#include "aggregate_fn_pushdown.hpp"

LogicalOperatorPtr TryPushdownAggregateFunctions(ClientContext &, LogicalOperatorPtr plan) {
    return plan;
}
