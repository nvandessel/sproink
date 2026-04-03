#ifndef SPROINK_H
#define SPROINK_H

#include <stdarg.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdlib.h>

typedef struct SproinkGraph SproinkGraph;

typedef struct SproinkPairs SproinkPairs;

typedef struct SproinkResults SproinkResults;

struct SproinkGraph *sproink_graph_build(uint32_t num_nodes,
                                         uint32_t num_edges,
                                         const uint32_t *sources,
                                         const uint32_t *targets,
                                         const double *weights,
                                         const uint8_t *kinds);

void sproink_graph_free(struct SproinkGraph *graph);

struct SproinkResults *sproink_activate(const struct SproinkGraph *graph,
                                        uint32_t num_seeds,
                                        const uint32_t *seed_nodes,
                                        const double *seed_activations,
                                        uint32_t max_steps,
                                        double decay_factor,
                                        double spread_factor,
                                        double min_activation,
                                        double sigmoid_gain,
                                        double sigmoid_center,
                                        bool inhibition_enabled,
                                        double inhibition_strength,
                                        uint32_t inhibition_breadth);

uint32_t sproink_results_len(const struct SproinkResults *results);

void sproink_results_nodes(const struct SproinkResults *results, uint32_t *out);

void sproink_results_activations(const struct SproinkResults *results, double *out);

void sproink_results_distances(const struct SproinkResults *results, uint32_t *out);

void sproink_results_free(struct SproinkResults *results);

struct SproinkPairs *sproink_extract_pairs(const struct SproinkResults *results,
                                           uint32_t num_seeds,
                                           const uint32_t *seed_nodes,
                                           double activation_threshold);

uint32_t sproink_pairs_len(const struct SproinkPairs *pairs);

void sproink_pairs_nodes(const struct SproinkPairs *pairs, uint32_t *out_a, uint32_t *out_b);

void sproink_pairs_activations(const struct SproinkPairs *pairs, double *out_a, double *out_b);

void sproink_pairs_free(struct SproinkPairs *pairs);

double sproink_oja_update(double current_weight,
                          double activation_a,
                          double activation_b,
                          double learning_rate,
                          double min_weight,
                          double max_weight);

#endif  /* SPROINK_H */
