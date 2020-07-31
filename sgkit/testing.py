from typing import Optional

import numpy as np

from .api import GenotypeCall, SgkitDataset
from .utils import split_array_chunks


def simulate_genotype_call_dataset(
    n_variant: int,
    n_sample: int,
    n_ploidy: int = 2,
    n_allele: int = 2,
    n_contig: int = 1,
    seed: Optional[int] = None,
) -> SgkitDataset[GenotypeCall]:
    """Simulate genotype calls and variant/sample data.

    Note that the data simulated by this function has no
    biological interpretation and that summary statistics
    or other methods applied to it will produce meaningless
    results.  This function is primarily a convenience on
    generating `Dataset` containers so quantities of interest
    should be overwritten, where appropriate, within the
    context of a more specific application.

    Parameters
    ----------
    n_variant : int
        Number of variants to simulate
    n_sample : int
        Number of samples to simulate
    n_ploidy : int
        Number of chromosome copies in each sample
    n_allele: int
        Number of alleles to simulate
    n_contig : int, optional
        Number of contigs to partition variants with,
        controlling values in `variant_contig`. Values
        will all be 0 by default with `n_contig` == 1.
    seed : int, optional
        Seed for random number generation

    Returns
    -------
    Dataset
        Dataset from `sgkit.create_genotype_call_dataset`.
    """
    rs = np.random.RandomState(seed=seed)
    call_genotype = rs.randint(
        0, n_allele, size=(n_variant, n_sample, n_ploidy), dtype=np.int8
    )
    contig_size = split_array_chunks(n_variant, n_contig)
    contig = np.repeat(np.arange(n_contig), contig_size)
    contig_names = np.unique(contig)
    position = np.concatenate([np.arange(contig_size[i]) for i in range(n_contig)])
    assert position.size == contig.size
    alleles = rs.choice(["A", "C", "G", "T"], size=(n_variant, n_allele)).astype("S")
    sample_id = np.array([f"S{i}" for i in range(n_sample)])
    return SgkitDataset.create_genotype_call_dataset(
        variant_contig_names=list(contig_names),
        variant_contig=contig,
        variant_position=position,
        variant_alleles=alleles,
        sample_id=sample_id,
        call_genotype=call_genotype,
    )
