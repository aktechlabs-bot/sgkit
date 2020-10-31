import logging

from sgkit.distance.api import pairwise_distance
import dask.array as da

from dask.distributed import Client
from dask.diagnostics import ProgressBar, ResourceProfiler
from bokeh.io import output_notebook

import numpy as np
import dask
import fsspec
import zarr


output_notebook()


def setup_logging():
    """Sets up general and dask logger"""
    ProgressBar().register()
    logging_format = "%(asctime)s %(levelname)9s %(lineno)4s %(module)s: %(message)s"
    logging.basicConfig(level=logging.INFO, format=logging_format)
    logging.info("Logging initialised")


def main():
    setup_logging()
    logging.info("Getting store object")
    #store = fsspec.get_mapper('gs://ag1000g-release/phase2.AR1/variation/main/zarr/all/ag1000g.phase2.ar1/')
    #callset_snps = zarr.open_consolidated(store=store)
    sgkit_data = zarr.open_group('/work/aktech/sgkit_data/output/')

    logging.info("Setting dask config")
    dask.config.set({'temporary_directory': '/work/aktech/tmp'})
    logging.info("Creating dask client")
    client = Client(n_workers=4,
                    threads_per_worker=2,
                    memory_limit='6GB')
    logging.info(f"Client created: {client}")

    logging.info("Getting genotype data")
    gt = sgkit_data['call_genotype']
    gt_da = da.from_zarr(gt, chunks=(-1, 1, -1))
    x = gt_da[:, :, 1].T
    x = x[:250]
    logging.info(f'The x is: {x}')
    logging.info("Starting the pairwise calculation")
    with ProgressBar(), ResourceProfiler() as prof:
        out = pairwise_distance(x)

    logging.info("Pairwise process complete")
    np.save('output.csv', out)
    logging.info("All done!")


if __name__ == '__main__':
    main()

