Timing analysis
---------------

Perform timing analysis of methods. For clusTCR, as requires its own environment, clustering is performed in compartor_tooling/clustCR/scripts.

For tcrdist, for purpose of timing we use the latest algorithm, tcrdist3, for calculating the distance matrix and perform dbscan clustering on this. This is appreciable faster than tcrdist original, so we limit the comparison to this as the original tcrdist is known to scale very poorly.

