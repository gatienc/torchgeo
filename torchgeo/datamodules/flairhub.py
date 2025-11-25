# Copyright (c) TorchGeo Contributors. All rights reserved.
# Licensed under the MIT License.

"""FLAIRHUB datamodule."""

from typing import Any

from torch.utils.data import Subset

from .geo import NonGeoDataModule
from .utils import group_shuffle_split


class FLAIRHUBDataModule(NonGeoDataModule):
    """LightningDataModule implementation for the FLAIRHUB dataset.

    Implements domain-based train/val/test splits to ensure that samples from
    the same domain stay together in the same split. This prevents data leakage
    between train, validation, and test sets.

    Default split ratios: 85% train, 10% validation, 5% test.

    .. versionadded:: 0.8
    """

    def __init__(
        self,
        batch_size: int = 64,
        num_workers: int = 0,
        train_split: float = 0.85,
        val_split: float = 0.10,
        test_split: float = 0.05,
        random_state: int = 0,
        **kwargs: Any,
    ) -> None:
        """Initialize a new FLAIRHUBDataModule instance.

        Args:
            batch_size: Size of each mini-batch.
            num_workers: Number of workers for parallel data loading.
            train_split: Proportion of domains to include in the train split.
            val_split: Proportion of domains to include in the validation split.
            test_split: Proportion of domains to include in the test split.
            random_state: Controls the random splits (passed as seed to
                numpy.random.Generator), set for reproducible splits.
            **kwargs: Additional keyword arguments passed to
                :class:`~torchgeo.datasets.FLAIRHUB`.

        Raises:
            ValueError: If train_split, val_split, and test_split do not sum to 1.0,
                or if any split is not in the range (0, 1).
        """
        if not (0 < train_split < 1 and 0 < val_split < 1 and 0 < test_split < 1):
            raise ValueError(
                'train_split, val_split, and test_split must be in the range (0, 1)'
            )

        if abs(train_split + val_split + test_split - 1.0) > 1e-6:
            raise ValueError(
                f'train_split ({train_split}) + val_split ({val_split}) + '
                f'test_split ({test_split}) must sum to 1.0'
            )

        from ..datasets import FLAIRHUB

        super().__init__(FLAIRHUB, batch_size, num_workers, **kwargs)

        self.train_split = train_split
        self.val_split = val_split
        self.test_split = test_split
        self.random_state = random_state

    def _extract_domain(self, mask_path: str) -> str:
        """Extract domain identifier from a mask file path.

        Args:
            mask_path: Path to the mask file.

        Returns:
            Domain identifier (e.g., 'D004').
        """
        parts = mask_path.split('_')
        for part in parts:
            if '-' in part and 'D' in part:
                domain_year_candidate = part.split('/')[-1]
                if domain_year_candidate.startswith('D'):
                    domain = domain_year_candidate.split('-')[0]
                    return domain
        raise ValueError(f"Could not extract domain from path: {mask_path}")

    def setup(self, stage: str) -> None:
        """Set up datasets.

        Args:
            stage: Either 'fit', 'validate', 'test', or 'predict'.
        """
        # Create the full dataset to extract domain information
        dataset = self.dataset_class(**self.kwargs)

        # Extract domains from all samples
        domains = []
        for sample in dataset.files:
            mask_path = str(sample['mask'])
            domain = self._extract_domain(mask_path)
            domains.append(domain)

        # First split: train vs (val+test)
        train_indices, val_test_indices = group_shuffle_split(
            domains,
            train_size=self.train_split,
            test_size=self.val_split + self.test_split,
            random_state=self.random_state,
        )

        if stage in ['fit', 'validate']:
            # Second split: val vs test from the remaining domains
            val_test_domains = [domains[i] for i in val_test_indices]
            # Calculate proportions for val/test split
            # val_split and test_split are proportions of total, but we need
            # proportions of the val_test subset
            val_test_total = self.val_split + self.test_split
            val_proportion = self.val_split / val_test_total
            test_proportion = self.test_split / val_test_total

            val_indices_subset, test_indices_subset = group_shuffle_split(
                val_test_domains,
                train_size=val_proportion,
                test_size=test_proportion,
                random_state=self.random_state,
            )

            # Map back to original indices
            val_indices = [val_test_indices[i] for i in val_indices_subset]
            test_indices = [val_test_indices[i] for i in test_indices_subset]

            self.train_dataset = Subset(dataset, train_indices)
            self.val_dataset = Subset(dataset, val_indices)

        if stage in ['test']:
            # For test stage, we need to compute val/test split again
            val_test_domains = [domains[i] for i in val_test_indices]
            val_test_total = self.val_split + self.test_split
            val_proportion = self.val_split / val_test_total
            test_proportion = self.test_split / val_test_total

            val_indices_subset, test_indices_subset = group_shuffle_split(
                val_test_domains,
                train_size=val_proportion,
                test_size=test_proportion,
                random_state=self.random_state,
            )

            test_indices = [val_test_indices[i] for i in test_indices_subset]
            self.test_dataset = Subset(dataset, test_indices)

