#include "spatial/index/rtree/rtree_index_create_physical.hpp"
#include "spatial/index/rtree/rtree_index.hpp"
#include "spatial/index/rtree/rtree_node.hpp"
#include "spatial/util/managed_collection.hpp"

#include "duckdb/catalog/catalog_entry/duck_index_entry.hpp"
#include "duckdb/catalog/catalog_entry/duck_table_entry.hpp"
#include "duckdb/catalog/catalog_entry/table_catalog_entry.hpp"
#include "duckdb/common/exception/transaction_exception.hpp"
#include "duckdb/storage/storage_manager.hpp"
#include "duckdb/storage/table_io_manager.hpp"
#include "duckdb/parallel/base_pipeline_event.hpp"

namespace duckdb {

unique_ptr<IndexBuildBindData> RtreeIndexBuildBind(IndexBuildBindInput &input) {
	return nullptr;
}

//-------------------------------------------------------------
// Global State
//-------------------------------------------------------------

class RTreeIndexBuildState final : public IndexBuildState {
public:
	//! Global index to be added to the table
	unique_ptr<RTreeIndex> rtree;

	//! The total number of leaf nodes in the RTree
	idx_t rtree_size = 0;
	idx_t slice_size = 0;
	idx_t rtree_level = 0;

	AllocatedData slice_buffer;

	//! The bottom most layer of the RTree
	ManagedCollection<RTreeEntry> curr_layer;
	ManagedCollection<RTreeEntry> next_layer;

	ManagedCollectionAppendState append_state;
	ManagedCollectionScanState scan_state;

	ManagedCollection<RTreeEntry> *curr_layer_ptr;
	ManagedCollection<RTreeEntry> *next_layer_ptr;

	RTreePointer current_pointer;

	idx_t entry_idx;
	idx_t max_node_capacity;

	explicit RTreeIndexBuildState(ClientContext &context)
	    : curr_layer(BufferManager::GetBufferManager(context)), next_layer(BufferManager::GetBufferManager(context)),
	      curr_layer_ptr(&next_layer), // We swap the order here so that it initializes properly later
	      next_layer_ptr(&curr_layer) {
	}
};

// init global state
unique_ptr<IndexBuildState> RtreeInitBuildState(IndexBuildInitStateInput &input) {
	auto gstate = make_uniq<RTreeIndexBuildState>(input.context);

	// Create the index
	auto &storage = input.table.GetStorage();
	auto &table_manager = TableIOManager::Get(storage);
	auto &constraint_type = input.info.constraint_type;
	auto &db = storage.db;
	gstate->rtree =
		make_uniq<RTreeIndex>(input.info.index_name, constraint_type, input.storage_ids, table_manager, input.expressions, db,
							  input.info.options, IndexStorageInfo(), input.estimated_cardinality);

	gstate->max_node_capacity = gstate->rtree->tree->GetConfig().max_node_capacity;
	gstate->entry_idx = gstate->max_node_capacity;

	gstate->curr_layer.InitializeAppend(gstate->append_state);

	return std::move(gstate);
}

class RtreeIndexBuildSinkState final : public IndexBuildSinkState {
public:
	unique_ptr<ColumnDataCollection> collection;
	ColumnDataAppendState append_state;
};

// init local sink state
unique_ptr<IndexBuildSinkState> RtreeInitSinkState(IndexBuildInitSinkInput &input) {
	auto state = make_uniq<RtreeIndexBuildSinkState>();
	// TODO: Get the type of the sink chunk somehow
	vector<LogicalType> data_types = {input.data_types[0], LogicalType::ROW_TYPE};
	state->collection = make_uniq<ColumnDataCollection>(BufferManager::GetBufferManager(input.context), data_types);
	state->collection->InitializeAppend(state->append_state);
	return std::move(state);
}

//-------------------------------------------------------------
// Sink
//-------------------------------------------------------------

void RtreeIndexBuildSink(ExecutionContext &context, DataChunk &chunk, OperatorSinkInput &input) {
	auto &gstate = input.global_state.Cast<RTreeIndexBuildState>();
	//
	// if (chunk.size() == 0) {
	// 	return SinkResultType::NEED_MORE_INPUT;
	// }

	// TODO: Dont flatten chunk
	chunk.Flatten();

	const auto &bbox_vecs = StructVector::GetEntries(chunk.data[0]);
	const auto &rowid_data = FlatVector::GetData<row_t>(chunk.data[1]);
	const auto min_x_data = FlatVector::GetData<float>(*bbox_vecs[0]);
	const auto min_y_data = FlatVector::GetData<float>(*bbox_vecs[1]);
	const auto max_x_data = FlatVector::GetData<float>(*bbox_vecs[2]);
	const auto max_y_data = FlatVector::GetData<float>(*bbox_vecs[3]);

	// Vectorized conversion from columnar to row-wise
	RTreeEntry entries[STANDARD_VECTOR_SIZE];
	for (idx_t elem_idx = 0; elem_idx < chunk.size(); elem_idx++) {
		auto &entry = entries[elem_idx];
		entry.pointer = RTree::MakeRowId(rowid_data[elem_idx]);
		entry.bounds.min.x = min_x_data[elem_idx];
		entry.bounds.min.y = min_y_data[elem_idx];
		entry.bounds.max.x = max_x_data[elem_idx];
		entry.bounds.max.y = max_y_data[elem_idx];
	}

	// Append the chunk to the current layer
	gstate.curr_layer.Append(gstate.append_state, entries, entries + chunk.size());

	// Count the number of entries
	gstate.rtree_size += chunk.size();

	// return SinkResultType::NEED_MORE_INPUT;
}

// not really necessary, this is just the global state
class RTreeIndexBuildWorkState final : public IndexBuildWorkState {
public:
	explicit RTreeIndexBuildWorkState(RTreeIndexBuildState &state) : state(state) {
	}

	RTreeIndexBuildState &state;
};

// not really necessary, this is just the global state
unique_ptr<IndexBuildWorkState> RTreeInitWorkState(IndexBuildInitWorkInput &input) {
	auto &gstate = input.global_state->Cast<RTreeIndexBuildState>();
	auto lstate = make_uniq<RTreeIndexBuildWorkState>(gstate);
	return std::move(lstate);
}

//-------------------------------------------------------------
// RTree Construction
//-------------------------------------------------------------
bool IndexBuildWork(IndexBuildWorkInput &input){
	auto &gstate = input.global_state->Cast<RTreeIndexBuildState>();
	auto &tree = gstate.rtree->tree;

	auto slice_begin = reinterpret_cast<RTreeEntry *>(gstate.slice_buffer.get());
	auto slice_end = slice_begin + gstate.slice_size;

	// Now, we have our base layer with all the leaves, we need to build the rest of the tree layer by layer
	while (gstate.curr_layer_ptr->Count() != 1) {
		if (gstate.scan_state.IsDone()) {

			// Swap the layers and initialize the next layer
			std::swap(gstate.curr_layer_ptr, gstate.next_layer_ptr);

			if (gstate.curr_layer_ptr->Count() == 1) {
				// We are done!
				break;
			}

			// Current layer size, divided by the node capacity (rounded up)
			const auto next_layer_size =
			    (gstate.curr_layer_ptr->Count() + gstate.max_node_capacity - 1) / gstate.max_node_capacity;
			gstate.next_layer_ptr->Clear();
			gstate.next_layer_ptr->InitializeAppend(gstate.append_state, next_layer_size);
			gstate.curr_layer_ptr->InitializeScan(gstate.scan_state, true);
		}

		idx_t child_idx = gstate.max_node_capacity;
		RTreePointer current_ptr;
		bool needs_insertion = false;

		auto scan_count = gstate.curr_layer_ptr->Scan(gstate.scan_state, slice_begin, slice_end);

		while (scan_count != 0) {
			// Sort the slice by the bounding box y-min value
			std::sort(slice_begin, slice_begin + scan_count, [&](const RTreeEntry &a, const RTreeEntry &b) {
				return a.bounds.Center().y < b.bounds.Center().y;
			});

			// Now we can build the next layer
			idx_t scan_idx = 0;
			while (scan_idx < scan_count) {

				// Initialize a new node if we have to
				if (child_idx == gstate.max_node_capacity) {
					auto node_type = gstate.rtree_level == 0 ? RTreeNodeType::LEAF_PAGE : RTreeNodeType::BRANCH_PAGE;
					current_ptr = tree->MakePage(node_type);
					child_idx = 0;
					needs_insertion = true;
				}

				const auto remaining_capacity = gstate.max_node_capacity - child_idx;
				const auto remaining_elements = scan_count - scan_idx;

				// Dereference the current node
				auto &node = tree->RefMutable(current_ptr);

				for (idx_t j = 0; j < MinValue<idx_t>(remaining_capacity, remaining_elements); j++) {
					node.PushEntry(slice_begin[scan_idx++]);
					child_idx++;
				}

				if (child_idx == gstate.max_node_capacity) {
					// Append the current node to the layer
					if (current_ptr.GetType() == RTreeNodeType::LEAF_PAGE) {
						// If the node is a leaf node, sort it by row id
						node.SortEntriesByRowId();
					}
					auto node_bounds = node.GetBounds();
					gstate.next_layer_ptr->Append(gstate.append_state, RTreeEntry {current_ptr, node_bounds});
					needs_insertion = false;

					node.Verify(gstate.max_node_capacity);
				}
			}

			// Scan the next batch
			scan_count = gstate.curr_layer_ptr->Scan(gstate.scan_state, slice_begin, slice_end);
		}

		// If the layer was exhausted before we filled the last node, we need to insert it now
		if (needs_insertion) {
			auto &node = tree->RefMutable(current_ptr);
			if (current_ptr.GetType() == RTreeNodeType::LEAF_PAGE) {
				// If the node is a leaf node, sort it by row id
				node.SortEntriesByRowId();
			}
			auto node_bounds = node.GetBounds();
			gstate.next_layer_ptr->Append(gstate.append_state, RTreeEntry {current_ptr, node_bounds});
			needs_insertion = false;
		}

		// We are done with this layer, pop it
		gstate.rtree_level++;

		// todo, return true?
		return true;
	}

	// Set the root node!
	auto root = gstate.curr_layer_ptr->Fetch(0);

	if (root.pointer.GetType() == RTreeNodeType::ROW_ID) {
		// Create a leaf node to hold this row id
		auto root_leaf_ptr = tree->MakePage(RTreeNodeType::LEAF_PAGE);
		auto &node = tree->RefMutable(root_leaf_ptr);
		node.PushEntry(RTreeEntry {root.pointer, root.bounds});
		tree->SetRoot(RTreeEntry {root_leaf_ptr, root.bounds});
	} else {
		// Else, just set the root node
		D_ASSERT(root.pointer.IsPage());
		auto root_entry = gstate.curr_layer_ptr->Fetch(0);
		tree->SetRoot(root_entry);
	}

	return true;
}

static void AddIndexToCatalog(ClientContext &context, CreateRTreeIndexGlobalState &gstate, CreateIndexInfo &info,
                              DuckTableEntry &table) {

	// Now actually add the index to the storage
	auto &storage = table.GetStorage();

	if (!storage.IsRoot()) {
		throw TransactionException("Cannot create index on non-root transaction");
	}

	// Create the index entry in the catalog
	auto &schema = table.schema;

	if (schema.GetEntry(schema.GetCatalogTransaction(context), CatalogType::INDEX_ENTRY, info.index_name)) {
		if (info.on_conflict != OnCreateConflict::IGNORE_ON_CONFLICT) {
			throw CatalogException("Index with name \"%s\" already exists", info.index_name);
		}
		// IF NOT EXISTS on existing index. We are done.
		// TODO: Early out before this.
		return;
	}

	const auto index_entry = schema.CreateIndex(schema.GetCatalogTransaction(context), info, table).get();
	D_ASSERT(index_entry);
	auto &duck_index = index_entry->Cast<DuckIndexEntry>();
	duck_index.initial_index_size = gstate.rtree->Cast<BoundIndex>().GetInMemorySize();

	// Finally add it to storage
	storage.AddIndex(std::move(gstate.rtree));
}

class RTreeIndexConstructionEvent final : public BasePipelineEvent {
public:
	RTreeIndexConstructionEvent(CreateRTreeIndexGlobalState &gstate_p, Pipeline &pipeline_p, CreateIndexInfo &info_p,
	                            DuckTableEntry &table_p, const PhysicalCreateRTreeIndex &op_p)
	    : BasePipelineEvent(pipeline_p), gstate(gstate_p), info(info_p), table(table_p), op(op_p) {
	}

	void Schedule() override {
		auto &context = pipeline->GetClientContext();

		// We only schedule 1 task, as the bottom-up construction is single-threaded.
		vector<shared_ptr<Task>> tasks;
		tasks.push_back(make_uniq<RTreeIndexConstructionTask>(shared_from_this(), context, gstate, op));
		SetTasks(std::move(tasks));
	}

	void FinishEvent() override {
		AddIndexToCatalog(pipeline->GetClientContext(), gstate, info, table);
	}

private:
	CreateRTreeIndexGlobalState &gstate;
	CreateIndexInfo &info;
	DuckTableEntry &table;
	const PhysicalCreateRTreeIndex &op;
};

//-------------------------------------------------------------
// Finalize
//-------------------------------------------------------------
SinkFinalizeType PhysicalCreateRTreeIndex::Finalize(Pipeline &pipeline, Event &event, ClientContext &context,
                                                    OperatorSinkFinalizeInput &input) const {
	auto &gstate = input.global_state.Cast<CreateRTreeIndexGlobalState>();
	info->column_ids = storage_ids;

	if (gstate.rtree_size == 0) {
		// No entries to build the RTree from, we are done
		AddIndexToCatalog(context, gstate, *info, table);
		return SinkFinalizeType::READY;
	}

	// Otherwise, we need to build the RTree

	// Calculate the vertical slice size
	// square root of the total number of entries divide by the capacity of a node, rounded up
	gstate.slice_size = ExactNumericCast<idx_t>(std::ceil(
	                        std::sqrt((gstate.rtree_size + gstate.max_node_capacity - 1) / gstate.max_node_capacity))) *
	                    gstate.max_node_capacity;

	// Allocate a buffer for the vertical slice
	// (this can get quite large, so we allocate it on the buffer manager)
	gstate.slice_buffer =
	    BufferManager::GetBufferManager(context).GetBufferAllocator().Allocate(gstate.slice_size * sizeof(RTreeEntry));

	// Schedule the construction of the RTree
	auto construction_event = make_uniq<RTreeIndexConstructionEvent>(gstate, pipeline, *info, table, *this);
	event.InsertEvent(std::move(construction_event));

	return SinkFinalizeType::READY;
}

} // namespace duckdb
