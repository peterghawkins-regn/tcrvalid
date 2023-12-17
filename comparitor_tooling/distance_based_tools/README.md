Tooling around distance-based tools: tcr-dist and iSMART
-------------------

tools to send data in and out of correct formatting and ingestion, re-mapping the clusters and so on.

Adaptions to tcr-dist to pull out the cluster definitions are in modified_third_party.

- this is not set up to be installed with tcrvalid
 - instead is a standaone additional tool for purpose of comparison only
 
Tools inside:
 - iSMART (with distance matrix export option)
 - tcr-dist original 
     - with cluster export as file
     - with optional additional radius setting for clustering algo
 - tcr-dist parallel
     - tcr-dist clustering using dbscan and parallelised distance calculation
     - n.b essentially legacy - one can use tcrdist3 if want speed - same process as here though.

### To Install for use as comparisons

run sh setup_external_tools.sh from this directory:

```console
(base) [username /path/to/distance_based_tools/]$ sh setup_external_tools.py
```

when running from py code or notebooks, use sys.path.append(/path/to/this/dir)

you will then be able to import from tcrclustering

## Example running

 - see example_distance_based_runs.ipynb for example of using these tools to cluster and score. 
 
 
