# Copyright (c) TorchGeo Contributors. All rights reserved.
# Licensed under the MIT License.

"""FLAIRHUB dataset."""

import glob
import json
import os
from collections.abc import Callable, Collection, Sequence
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, ClassVar, cast, Literal
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import rasterio
import torch
from matplotlib.colors import BoundaryNorm, ListedColormap
from matplotlib.figure import Figure
from matplotlib.patches import Patch, Rectangle
from PIL import Image
from torch import Tensor

from .errors import DatasetNotFoundError, RGBBandsMissingError
from .geo import GeoDataset, NonGeoDataset
from .utils import check_integrity, download_url, extract_archive


def min_max_normalize_plot(image: np.ndarray) -> np.ndarray:
    """Normalize image for plotting by scaling to [0, 1].
    
    Args:
        image: Input image array
        
    Returns:
        Normalized image array in range [0, 1]
    """
    image_min = image.min()
    image_max = image.max()
    if image_max - image_min == 0:
        return np.zeros_like(image)
    return (image - image_min) / (image_max - image_min)



class FLAIRHUB(NonGeoDataset):
    """FLAIR-HUB: Large-scale Multimodal Dataset for Land Cover and Crop Mapping dataset.

    FLAIR-HUB (<https://github.com/IGNF/FLAIR-HUB>) builds upon and includes the FLAIR#1 and FLAIR#2 datasets, expanding them into a unified, large-scale, multi-sensor land-cover resource with very-high-resolution annotations. Spanning over 2,500 km² of diverse French ecoclimatic regions and landscapes, FLAIR-HUB features 63 billion hand-annotated pixels across 19 land-cover and 23 crop type classes.

    The dataset integrates complementary sources including aerial imagery, SPOT and Sentinel satellite acquisitions, surface models, and historical aerial photographs. This offers rich spatial, spectral, and temporal diversity, supporting a broad range of research tasks, including semantic segmentation, multimodal fusion, and self-supervised learning. FLAIR-HUB is designed as a continuously growing resource, with new modalities and annotations to be released in future updates.

    Dataset features:

    ROI / Area Covered: 2,822 ROIs / 2,528 km²  
    Departments (France): 74  
    AI Patches (512x512 px): 241,100  
    Annotated Pixels: 63.2 billion  
    Sentinel-2 Acquisitions: 256,221  
    Sentinel-1 Acquisitions: 532,696  
    Total Files: ~2.5 million  
    Total Dataset Size: ~750 GB  

    Dataset structure:
    
    The dataset is organized by domains (geographical areas) and years. Each domain-year
    combination has multiple modalities available for download. The full dataset contains
    66 unique domains (D004-D091, non-consecutive), with years ranging from 2017-2022.
    Most domains have data for a single year, but some have multiple years available.
    
    Available modalities (100% coverage across all domains):
        - AERIAL_RGBI: High-resolution aerial imagery (RGB + NIR, 0.2m)
        - SPOT_RGBI: SPOT satellite imagery (RGB + NIR, 1.5m)
        - DEM_ELEV: Digital Elevation Model (DSM + DTM, 1m)
        - AERIAL-RLT_PAN: Historical aerial panchromatic (1950s)
        - SENTINEL1-ASC_TS: Sentinel-1 SAR Ascending time series (VV + VH)
        - SENTINEL1-DESC_TS: Sentinel-1 SAR Descending time series (VV + VH)
        - SENTINEL2_TS: Sentinel-2 multispectral time series (12 bands, 10m)
        - SENTINEL2_MSK-SC: Sentinel-2 scene classification mask
        - AERIAL_LABEL-COSIA: Land cover labels (19 classes)
        - ALL_LABEL-LPIS: Crop type labels (23 classes)
    
    Example domains: D004, D005, D006, D017 (multi-year), D033 (multi-year), ...
    Available years: 2017, 2018, 2019, 2020, 2021, 2022
    
    Automatic download:
        Set download=True to automatically download requested modalities from HuggingFace.
        Only the modalities you select will be downloaded, saving bandwidth and storage.

    Dataset classes:
        AERIAL_LABEL-COSIA (Land Cover):
            0:  urban
            1:  greenhouse
            2:  swimming_pool
            3:  impervious surfaces
            4:  pervious surface
            5:  bare soil
            6:  water
            7:  snow
            8:  herbaceous vegetation
            9:  agricultural land
            10: plowed land
            11: vineyard
            12: deciduous
            13: coniferous
            14: brushwood
            15: clear cut
            16: ligneous
            17: mixed
            18: undefined

        ALL_LABEL-LPIS (Crop Type):
            0: grasses
            1: wheat
            2: barley
            3: maize
            4: other cereals
            5: rice
            6: flax/hemp/tobacco
            7: sunflower
            8: rapeseed
            9: other oilseed crops
            10: soy
            11: other protein crops
            12: fodder legumes
            13: beetroots
            14: potatoes
            15: other arable crops
            16: vineyard
            17: olive groves
            18: fruits orchards
            19: nut orchards
            20: other permanent crops
            21: mixed crops
            22: background
        

    If you use this dataset in your research, please cite the following paper:

    * https://arxiv.org/abs/2506.07080
    """

    # AERIAL-RLT_PAN (Historical Aerial Panchromatic)
    aerial_rlt_pan_bands: tuple[str] = ('PAN',)

    # AERIAL_RGBI
    aerial_rgb_bands: tuple[str, str, str] = ('R', 'G', 'B')
    aerial_all_bands: tuple[str, str, str, str] = ('R', 'G', 'B', 'NIR')

    # SPOT_RGBI
    spot_all_bands: tuple[str, str, str, str] = ('R', 'G', 'B', 'NIR')
    spot_rgb_bands: tuple[str, str, str] = ('R', 'G', 'B')

    # DEM_ELEV (Digital Elevation Model)
    dem_elev_bands: tuple[str, str] = ('DSM', 'DTM')

    # SENTINEL1-ASC_TS / SENTINEL1-DESC_TS (Time Series)
    sentinel1_bands: tuple[str, str] = ('VV', 'VH')

    # SENTINEL2_TS (Time Series)
    sentinel2_rgb_bands: tuple[str, str, str] = ('B04', 'B03', 'B02')
    sentinel2_all_bands: tuple[str, ...] = (
        'B01',
        'B02',
        'B03',
        'B04',
        'B05',
        'B06',
        'B07',
        'B08',
        'B8A',
        'B09',
        'B11',
        'B12',
    )

    # SENTINEL2_MSK-SC (Scene Classification Mask)
    sentinel2_msk_bands: tuple[str] = ('SCL',)
    
    # HuggingFace dataset URL
    url_base: ClassVar[str] = 'https://huggingface.co/datasets/IGNF/FLAIR-HUB/resolve/main/data'
    
    # Available domain-year combinations
    # Format: {domain: [years]}
    # Note: Some domains have multiple years available
    domain_years: ClassVar[dict[str, list[str]]] = {
        'D004': ['2021'],
        'D005': ['2018'],
        'D006': ['2020'],
        'D007': ['2020'],
        'D008': ['2019'],
        'D009': ['2019'],
        'D010': ['2019'],
        'D011': ['2021'],
        'D012': ['2019'],
        'D013': ['2020'],
        'D014': ['2020'],
        'D015': ['2020'],
        'D016': ['2020'],
        'D017': ['2018', '2021'],  # Multiple years available
        'D018': ['2020'],
        'D020': ['2019'],
        'D021': ['2020'],
        'D022': ['2021'],
        'D023': ['2020'],
        'D026': ['2020'],
        'D029': ['2021'],
        'D030': ['2021'],
        'D031': ['2019'],
        'D032': ['2019'],
        'D033': ['2018', '2021'],  # Multiple years available
        'D034': ['2021'],
        'D035': ['2020'],
        'D036': ['2020'],
        'D037': ['2021'],
        'D038': ['2021'],
        'D040': ['2021'],
        'D041': ['2021'],
        'D044': ['2020', '2022'],  # Multiple years available
        'D045': ['2020'],
        'D046': ['2019'],
        'D049': ['2020'],
        'D051': ['2019'],
        'D052': ['2019'],
        'D055': ['2018'],
        'D056': ['2019'],
        'D058': ['2020'],
        'D060': ['2021'],
        'D061': ['2020'],
        'D063': ['2019'],
        'D064': ['2021'],
        'D065': ['2019'],
        'D066': ['2021'],
        'D067': ['2021'],
        'D068': ['2021'],
        'D069': ['2020'],
        'D070': ['2020'],
        'D071': ['2020'],
        'D072': ['2019'],
        'D073': ['2022'],
        'D074': ['2020'],
        'D075': ['2021'],
        'D076': ['2019'],
        'D077': ['2021'],
        'D078': ['2021'],
        'D080': ['2017', '2021'],  # Multiple years available
        'D081': ['2020'],
        'D083': ['2020'],
        'D084': ['2021'],
        'D085': ['2019'],
        'D086': ['2020'],
        'D091': ['2021'],
    }
    
    # Modality mapping: (use_flag_attribute, directory_suffix, is_historical)
    modality_map: ClassVar[list[tuple[str, str, bool]]] = [
        ('use_aerial_rgbi', 'AERIAL_RGBI', False),
        ('use_spot_rgbi', 'SPOT_RGBI', False),
        ('use_dem_elev', 'DEM_ELEV', False),
        ('use_sentinel1_asc_ts', 'SENTINEL1-ASC_TS', False),
        ('use_sentinel1_desc_ts', 'SENTINEL1-DESC_TS', False),
        ('use_sentinel2_ts', 'SENTINEL2_TS', False),
        ('use_sentinel2_msk_sc', 'SENTINEL2_MSK-SC', False),
        ('use_aerial_rlt_pan', 'AERIAL-RLT_PAN', True),  # Historical (195X)
    ]

    # Note: the original dataset contains 19 classes, but the dataset paper suggests not using the 3 last classes as they are nearly empty
    cosia:dict[str, Any] = {
        "classes" : [
            "building",
            "greenhouse",
            "swimming_pool",
            "impervious surface",
            "pervious surface",
            "bare soil",
            "water",
            "snow",
            "herbaceous vegetation",
            "agricultural land",
            "plowed land",
            "vineyard",
            "deciduous",
            "coniferous",
            "brushwood",
            "clear cut",
            "ligneous",
            "mixed",
            "undefined",
            ],
        # Define a colormap for the classes
        "cmap" : ListedColormap(
            [
            '#d3d3d3',  # building
            '#a9a9a9',  # greenhouse
            '#00bfff',  # swimming_pool
            '#696969',  # impervious surface
            '#d2b48c',  # pervious surface
            '#8b4513',  # bare soil
            '#0000ff',  # water
            '#ffffff',  # snow
            '#90ee90',  # herbaceous vegetation
            '#ffd700',  # agricultural land
            '#cd853f',  # plowed land
            '#9370db',  # vineyard
            '#228b22',  # deciduous
            '#006400',  # coniferous
            '#8fbc8f',  # brushwood
            '#ff4500',  # clear cut
            '#556b2f',  # ligneous
            '#2e8b57',  # mixed
            '#808080',  # undefined
            ]
        )}
    lpis = {
        "classes" : [
            "grasses",
            "wheat",
            "barley",
            "maize",
            "other cereals",
            "rice",
            "flax/hemp/tobacco",
            "sunflower",
            "rapeseed",
            "other oilseed crops",
            "soy",
            "other protein crops",
            "fodder legumes",
            "beetroots",
            "potatoes",
            "other arable crops",
            "vineyard",
            "olive groves",
            "fruits orchards",
            "nut orchards",
            "other permanent crops",
            "mixed crops",
            "background"
        ],
        "cmap" : ListedColormap(
            [
                '#8dd3c7',  # grasses
                '#ffffb3',  # wheat
                '#bebada',  # barley
                '#fb8072',  # maize
                '#80b1d3',  # other cereals
                '#fdb462',  # rice
                '#b3de69',  # flax/hemp/tobacco
                '#fccde5',  # sunflower
                '#d9d9d9',  # rapeseed
                '#bc80bd',  # other oilseed crops
                '#ccebc5',  # soy
                '#ffed6f',  # other protein crops
                '#a6cee3',  # fodder legumes
                '#1f78b4',  # beetroots
                '#b2df8a',  # potatoes
                '#33a02c',  # other arable crops
                '#fb9a99',  # vineyard
                '#e31a1c',  # olive groves
                '#fdbf6f',  # fruits orchards
                '#ff7f00',  # nut orchards
                '#cab2d6',  # other permanent crops
                '#6a3d9a',  # mixed crops
                '#000000',  # background
            ])}
    

    def __init__(
        self,
        root: Path = Path('data'),
        transforms: Callable[[dict[str, Tensor]], dict[str, Tensor]] | None = None,
        download: bool = False,
        checksum: bool = False,
        use_aerial_rgbi: bool = False,
        use_aerial_rlt_pan: bool = False,
        use_dem_elev: bool = False,
        use_spot_rgbi: bool = False,
        use_sentinel2_ts: bool = False,
        use_sentinel1_asc_ts: bool = False,
        use_sentinel1_desc_ts: bool = False,
        dataset_type: Literal["land_cover", "crop_type"] = "land_cover",
    ) -> None:
        """Initialize a new FLAIRHUB dataset instance.

        The FLAIR-HUB dataset provides multiple complementary data modalities for land
        cover and crop type mapping. You can selectively load any combination of the
        available modalities using the corresponding boolean flags.

        Args:
            root: Root directory where dataset can be found or will be downloaded.
            transforms: Optional transforms to apply to samples.
            download: If True, download the dataset if it is not found.
            checksum: If True, verify the integrity of downloaded files using MD5 checksums.
            use_aerial_rgbi: If True, load high-resolution aerial imagery (RGB + NIR, 0.2m resolution).
            use_aerial_rlt_pan: If True, load historical aerial panchromatic imagery from 1950s.
            use_dem_elev: If True, load Digital Elevation Model data (DSM + DTM, 1m resolution).
            use_spot_rgbi: If True, load SPOT satellite imagery (RGB + NIR, 1.5m resolution).
            use_sentinel2_ts: If True, load Sentinel-2 multispectral time series (12 bands, 10m resolution).
            use_sentinel1_asc_ts: If True, load Sentinel-1 SAR Ascending time series (VV + VH polarizations).
            use_sentinel1_desc_ts: If True, load Sentinel-1 SAR Descending time series (VV + VH polarizations).
            dataset_type: Type of labels to use. Choose 'land_cover' for 19-class COSIA labels
                or 'crop_type' for 23-class LPIS crop classification labels.

        Raises:
            DatasetNotFoundError: If dataset is not found and *download* is False.
            ValueError: If *dataset_type* is not 'land_cover' or 'crop_type'.

        Note:
            At least one data modality must be enabled (set to True) to load samples.
            The dataset returns file paths for each enabled modality, allowing flexible
            data loading strategies based on your use case.

        Examples:
            >>> # Load aerial imagery with land cover labels
            >>> dataset = FLAIRHUB(root='data', use_aerial_rgbi=True, dataset_type='land_cover')
            >>> 
            >>> # Load multiple modalities for multi-modal fusion
            >>> dataset = FLAIRHUB(
            ...     root='data',
            ...     use_aerial_rgbi=True,
            ...     use_spot_rgbi=True,
            ...     use_dem_elev=True,
            ...     dataset_type='land_cover'
            ... )
            >>> 
            >>> # Access a sample (returns paths to data files)
            >>> sample = dataset[0]
            >>> print(sample.keys())  # dict_keys(['mask', 'aerial_rgbi', 'spot_rgbi', 'dem_elev'])

        .. versionadded:: 0.7
        """
        
        self.root = Path(root)
        self.transforms = transforms
        self.download = download
        self.checksum = checksum
        self.dataset_type = dataset_type
        
        # Store which modalities to use
        self.use_aerial_rgbi = use_aerial_rgbi
        self.use_aerial_rlt_pan = use_aerial_rlt_pan
        self.use_dem_elev = use_dem_elev
        self.use_spot_rgbi = use_spot_rgbi
        self.use_sentinel2_ts = use_sentinel2_ts
        self.use_sentinel2_msk_sc = False  # Not yet supported in init params
        self.use_sentinel1_asc_ts = use_sentinel1_asc_ts
        self.use_sentinel1_desc_ts = use_sentinel1_desc_ts
        
        # Verify at least one modality is selected
        modalities_enabled = any([
            use_aerial_rgbi, use_aerial_rlt_pan, use_dem_elev, use_spot_rgbi,
            use_sentinel2_ts, use_sentinel1_asc_ts, use_sentinel1_desc_ts
        ])
        if not modalities_enabled:
            raise ValueError("At least one data modality must be enabled")
        
        self._verify()
        self.files = self._load_files()

    def get_num_bands(self, include_sentinel_bands: bool = False) -> int:
        """Return the number of bands in the dataset.

        Returns:
            int: number of bands in the initialized dataset (might vary from all_bands)
        """
        return (
            len(self.aerial_bands)
            if not include_sentinel_bands
            else len(self.aerial_bands) + len(self.sentinel_bands)
        )

    def __getitem__(self, index: int) -> dict[str, Tensor]:
        """Return an index within the dataset.

        Args:
            index: index to return

        Returns:
            dictionary containing tensors for each modality
            keys are the modality names : 'mask', 'aerial_rgbi', 'aerial_rlt_pan', 'spot_rgbi', 'dem_elev', 'sentinel2_ts', 'sentinel1_asc_ts', 'sentinel1_desc_ts'
        """
        
        sample = self.files[index]
        if self.transforms is not None:
            sample = self.transforms(sample)
        return sample

    def __len__(self) -> int:
        """Return the number of datapoints in the dataset.

        Returns:
            length of dataset
        """
        return len(self.files)


    def _get_requested_modalities(self) -> list[tuple[str, str, bool]]:
        """Get list of requested modalities based on flags.
        
        Returns:
            List of tuples (flag_name, modality_suffix, is_historical)
        """
        requested = []
        for flag_attr, modality_suffix, is_historical in self.modality_map:
            if getattr(self, flag_attr, False):
                requested.append((flag_attr, modality_suffix, is_historical))
        return requested
    
    def _get_label_modality(self) -> str:
        """Get the label modality directory name based on dataset type.
        
        Returns:
            Label directory name
        """
        if self.dataset_type == "land_cover":
            return "AERIAL_LABEL-COSIA"
        elif self.dataset_type == "crop_type":
            return "ALL_LABEL-LPIS"
        else:
            raise ValueError(f"Unknown dataset_type: {self.dataset_type}")
    
    def _load_files(self) -> list[dict[str, Path]]:
        """Load paths to all files for each sample in the dataset.
        
        This method scans the root directory for label files and builds
        dictionaries mapping modality names to file paths for each sample.

        Returns:
            List of dictionaries, one per sample. Each dictionary contains
            paths to the mask and requested modality files.
        """
        files_list = []
        label_modality = self._get_label_modality()
        requested_modalities = self._get_requested_modalities()
        
        # Find all label files across all domains
        pattern = f'*/*_{label_modality}/*/*.tif'
        label_paths = list(self.root.glob(pattern))
        

        if not label_paths:
            raise FileNotFoundError(
                f"No label files found in {self.root}. "
                f"Pattern searched: {pattern}"
            )
        
        # Build file dictionaries for each sample
        for label_path in label_paths:
            file_dict: dict[str, Path] = {}
            file_dict["mask"] = label_path
            
            # Extract domain-year from label path
            label_str = str(label_path)
            parts = label_str.split('_')
            
            # Find the part that contains domain-year pattern (Dxxx-yyyy)
            domain_year = None
            for part in parts:
                if '-' in part and 'D' in part:
                    # Extract just the domain-year part
                    domain_year_candidate = part.split('/')[-1]
                    if domain_year_candidate.startswith('D'):
                        domain_year = domain_year_candidate
                        break
            
            if not domain_year:
                print(f"Warning: Could not extract domain-year from {label_path}, skipping")
                continue
            
            domain = domain_year.split('-')[0]
            year = domain_year.split('-')[1]
            
            # Add each requested modality
            all_modalities_exist = True
            for flag_attr, modality_suffix, is_historical in requested_modalities:
                if is_historical:
                    # Historical data uses 195X instead of actual year
                    modality_path_str = label_str.replace(
                        f"{domain_year}_{label_modality}",
                        f"{domain}-195X_{modality_suffix}"
                    )
                else:
                    # Regular modalities use the same year
                    modality_path_str = label_str.replace(
                        f"{domain_year}_{label_modality}",
                        f"{domain_year}_{modality_suffix}"
                    )
                
                modality_path = Path(modality_path_str)
                
                if not modality_path.exists():
                    print(f"Warning: {modality_suffix} file not found: {modality_path}")
                    all_modalities_exist = False
                    break
                
                # Map to simplified key name (e.g., 'aerial_rgbi', 'spot_rgbi')
                key_name = flag_attr.replace('use_', '')
                file_dict[key_name] = modality_path
            
            # Only add sample if all requested modalities exist
            if all_modalities_exist:
                files_list.append(file_dict)
        
        if not files_list:
            raise FileNotFoundError(
                f"No complete samples found with all requested modalities in {self.root}"
            )
        
        print(f"Loaded {len(files_list)} samples with all requested modalities")
        return files_list

    def _load_image(self, path: Path) -> Tensor:
        """Load a single image.

        Args:
            path: path to the image

        Returns:
            tensor: the loaded image
        """
        with rasterio.open(path) as f:
            array: np.typing.NDArray[np.int_] = f.read()
            tensor = torch.from_numpy(array).float() / 255

        # Extract the bands of interest
        tensor = tensor[[int(band[-2:]) - 1 for band in self.aerial_bands]]

        if 'B05' in self.aerial_bands:
            # Height channel will always be the last dimension
            tensor[-1] = torch.div(tensor[-1], 5)

        return tensor

    def _load_sentinel(self, path: Path) -> Tensor:
        """Load a sentinel array.

        Args:
            path: path to sentinel img (data or snow cloud mask)

        Returns:
            tensor: image as tensors of shape TxCxHxW (time, channels, height, width)
        """
        tensor = torch.from_numpy(np.load(path)).float()
        return tensor[:, [int(band[-2:]) - 1 for band in self.sentinel_bands]]

    def _load_target(self, path: Path) -> Tensor:
        """Load a single mask corresponding to image.

        Args:
            path: path to the mask

        Returns:
            tensor: the mask of the image
        """
        with rasterio.open(path) as f:
            array: np.typing.NDArray[np.int_] = f.read(1)
            tensor = torch.from_numpy(array).long()
            # According to datapaper, the dataset contains classes beyond 13
            # however, those are grouped into a single "other" class
            # Rescale the classes to be in the range [0, 12] by subtracting 1
            torch.clamp(tensor - 1, 0, len(self.classes) - 1, out=tensor)

        return tensor

    def _verify(self) -> None:
        """Verify dataset integrity and download missing files.
        
        This method checks if the requested modalities are present for all
        domain-year combinations. If any are missing, it downloads them
        if download=True is set.
        """
        label_modality = self._get_label_modality()
        requested_modalities = self._get_requested_modalities()
        
        # Track which files need to be downloaded/extracted
        to_download: list[tuple[str, str, str]] = []  # (domain, year, modality)
        to_extract: list[tuple[str, str, str]] = []   # (domain, year, modality)
        
        # Check each domain-year combination
        for domain, years in self.domain_years.items():
            for year in years:
                domain_year = f"{domain}-{year}"
                
                # Always need labels
                modalities_to_check = [(None, label_modality, False)] + requested_modalities
                
                for flag_attr, modality_suffix, is_historical in modalities_to_check:
                    # Determine the actual domain-year string for this modality
                    if is_historical:
                        modality_domain_year = f"{domain}-195X"
                    else:
                        modality_domain_year = domain_year
                    
                    modality_dir = f"{modality_domain_year}_{modality_suffix}"
                    modality_path = self.root / modality_dir
                    modality_zip = self.root / f"{modality_dir}.zip"
                    
                    # Check if directory exists and has files
                    if modality_path.is_dir():
                        # Check if directory has .tif files
                        tif_files = list(modality_path.rglob('*.tif'))
                        if tif_files:
                            continue  # Already extracted and has data
                    
                    # Check if zip exists but not extracted
                    if modality_zip.is_file():
                        to_extract.append((domain, year if not is_historical else '195X', modality_suffix))
                    else:
                        # Need to download
                        to_download.append((domain, year if not is_historical else '195X', modality_suffix))
        
        # Extract any zips that exist but haven't been extracted
        if to_extract:
            print(f"Extracting {len(to_extract)} modality archives...")
            with ThreadPoolExecutor(max_workers=4) as executor:
                futures = {
                    executor.submit(self._extract, domain, year, modality): (domain, year, modality)
                    for domain, year, modality in to_extract
                }
                for future in as_completed(futures):
                    domain, year, modality = futures[future]
                    try:
                        future.result()
                    except Exception as e:
                        print(f"Error extracting {domain}-{year}_{modality}: {e}")
                        raise
        
        # Download any missing files
        if to_download:
            if not self.download:
                print(f"Missing {len(to_download)} modality archives. Set download=True to download them.")
                raise DatasetNotFoundError(self)
            
            with ThreadPoolExecutor(max_workers=4) as executor:
                futures = {
                    executor.submit(self._download_and_extract, domain, year, modality): (domain, year, modality)
                    for domain, year, modality in to_download
                }
                for future in as_completed(futures):
                    domain, year, modality = futures[future]
                    try:
                        future.result()
                    except Exception as e:
                        print(f"Error downloading/extracting {domain}-{year}_{modality}: {e}")
                        raise
        
        if not to_download and not to_extract:
            print('All requested modalities are already downloaded and extracted.')

    def _download(self, domain: str, year: str, modality: str) -> None:
        """Download a specific modality file from HuggingFace.
        
        Args:
            domain: Domain identifier (e.g., 'D004')
            year: Year or '195X' for historical data
            modality: Modality suffix (e.g., 'AERIAL_RGBI')
        """
        filename = f"{domain}-{year}_{modality}.zip"
        url = f"{self.url_base}/{filename}"
        
        download_url(
            url,
            str(self.root),
            filename=filename,
            md5=None,  # No checksums available
        )

    def _download_and_extract(self, domain: str, year: str, modality: str) -> None:
        """Download and extract a specific modality file from HuggingFace.
        
        Args:
            domain: Domain identifier (e.g., 'D004')
            year: Year or '195X' for historical data
            modality: Modality suffix (e.g., 'AERIAL_RGBI')
        """
        self._download(domain, year, modality)
        self._extract(domain, year, modality)

    def _extract(self, domain: str, year: str, modality: str) -> None:
        """Extract a specific modality archive and delete the zip file.
        
        Args:
            domain: Domain identifier (e.g., 'D004')
            year: Year or '195X' for historical data
            modality: Modality suffix (e.g., 'AERIAL_RGBI')
        """
        filename = f"{domain}-{year}_{modality}.zip"
        zipfile_path = self.root / filename
        
        if not zipfile_path.is_file():
            raise FileNotFoundError(f"Archive not found: {zipfile_path}")
        
        extract_archive(str(zipfile_path), str(self.root))
        
        zipfile_path.unlink()
       
    def _plot_mask(self, mask: np.ndarray, ax: plt.Axes, show_legend: bool = True) -> None:
        """Plot a label mask with appropriate colormap.
        
        Args:
            mask: Label mask array from rasterio (C, H, W) or (H, W)
            ax: Matplotlib axes to plot on
            show_legend: Whether to show the legend
        """
        if self.dataset_type == "crop_type":
            class_names = self.lpis["classes"]
            cmap = self.lpis["cmap"]
        elif self.dataset_type == "land_cover":
            class_names = self.cosia["classes"]
            cmap = self.cosia["cmap"]
        else:
            raise ValueError(f"Unknown dataset type: {self.dataset_type}")
        

        mask = mask[0]  # Take first channel: (1, H, W) -> (H, W)
        
        n_classes = len(class_names)
        bounds = np.arange(n_classes + 1) - 0.5
        norm = BoundaryNorm(bounds, n_classes)
        ax.imshow(mask, cmap=cmap, norm=norm)
        ax.set_title("Label Mask")
        ax.axis('off')
        
        if show_legend:
            present_classes = np.unique(mask)
            legend_elements = [
                Patch(facecolor=cmap(i), edgecolor='k', label=class_names[i])
                for i in present_classes if i < len(class_names)
            ]
            ax.legend(
                handles=legend_elements,
                bbox_to_anchor=(1.05, 1),
                loc='upper left',
                borderaxespad=0.,
                fontsize='small'
            )

    def _plot_rgb_modality(
        self, 
        data: np.ndarray, 
        ax: plt.Axes, 
        title: str,
        rgb_indices: list[int]
    ) -> None:
        """Plot an RGB image from multi-band data.
        
        Args:
            data: Multi-band image data (C, H, W) from rasterio
            ax: Matplotlib axes to plot on
            title: Title for the subplot
            rgb_indices: Indices of RGB bands
        """
        # Select RGB bands and transpose from (C, H, W) to (H, W, C) for matplotlib
        rgb_image = data[rgb_indices]  # Shape: (3, H, W)
        rgb_image = np.transpose(rgb_image, (1, 2, 0))  # Shape: (H, W, 3)
        rgb_image = min_max_normalize_plot(rgb_image)
        ax.imshow(rgb_image)
        ax.set_title(title)
        ax.axis('off')

    def _plot_grayscale_modality(
        self, 
        data: np.ndarray, 
        ax: plt.Axes, 
        title: str
    ) -> None:
        """Plot a single-band or grayscale image.
        
        Args:
            data: Single-band image data (H, W) or (1, H, W)
            ax: Matplotlib axes to plot on
            title: Title for the subplot
        """
        if data.ndim == 3:
            data = data[0]  # Take first band if multi-dimensional
        data_norm = min_max_normalize_plot(data)
        ax.imshow(data_norm, cmap='gray')
        ax.set_title(title)
        ax.axis('off')
    
    def _plot_dem(self, data: np.ndarray, ax: plt.Axes, title: str) -> None:
        """Plot DEM elevation data.
        
        Args:
            data: DEM data from rasterio (2, H, W) - DSM and DTM in (C, H, W) format
            ax: Matplotlib axes to plot on
            title: Title for the subplot
        """
        dem_band = data[0] if data.ndim == 3 else data  # Handle both (2, H, W) and (H, W)
        dem_norm = min_max_normalize_plot(dem_band)
        im = ax.imshow(dem_norm, cmap='terrain')
        ax.set_title(title)
        ax.axis('off')
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    def plot(
        self,
        sample: dict[str, Path],
        suptitle: str | None = None,
    ) -> Figure:
        """Plot a sample from the dataset.

        Args:
            sample: a sample returned by :meth:`__getitem__`
            suptitle: optional suptitle to use for figure

        Returns:
            a matplotlib Figure with the rendered sample
        """
        # Collect all available modalities to plot
        plot_data: dict[str, dict[str, Any]] = {}
        
        # Always plot the mask
        if 'mask' in sample:
            with rasterio.open(sample["mask"]) as f:
                mask = f.read()
            plot_data['mask'] = {
                'plot_type': 'mask',
                'data': mask,
                'title': 'mask'
            }
        
        # Plot aerial RGBI if available
        if 'aerial_rgbi' in sample:
            with rasterio.open(sample["aerial_rgbi"]) as f:
                aerial_rgbi = f.read()
            rgb_indices = [0, 1, 2]  # R, G, B bands
            plot_data['aerial_rgbi'] = {
                'plot_type': 'rgb',
                'data': aerial_rgbi,
                'title': 'Aerial RGBI',
                'rgb_indices': rgb_indices
            }
        
        # Plot SPOT RGBI if available
        if 'spot_rgbi' in sample:
            with rasterio.open(sample["spot_rgbi"]) as f:
                spot_rgbi = f.read()
            rgb_indices = [0, 1, 2]  # R, G, B bands
            plot_data['spot_rgbi'] = {
                'plot_type': 'rgb',
                'data': spot_rgbi,
                'title': 'SPOT RGBI',
                'rgb_indices': rgb_indices
            }
        
        # Plot historical aerial panchromatic if available
        if 'aerial_rlt_pan' in sample:
            with rasterio.open(sample["aerial_rlt_pan"]) as f:
                aerial_rlt = f.read()
            plot_data['aerial_rlt_pan'] = {
                'plot_type': 'grayscale',
                'data': aerial_rlt,
                'title': 'Historical Aerial'
            }
        
        # Plot DEM if available
        if 'dem_elev' in sample:
            with rasterio.open(sample["dem_elev"]) as f:
                dem = f.read()
            plot_data['dem_elev'] = {
                'plot_type': 'dem',
                'data': dem,
                'title': 'DEM Elevation'
            }
        
        # Plot Sentinel-1 Ascending if available
        if 'sentinel1_asc_ts' in sample:
            with rasterio.open(sample["sentinel1_asc_ts"]) as f:
                s1_asc = f.read()
            plot_data['sentinel1_asc_ts'] = {
                'plot_type': 'grayscale',
                'data': s1_asc,
                'title': 'Sentinel-1 ASC'
            }
        
        # Plot Sentinel-1 Descending if available
        if 'sentinel1_desc_ts' in sample:
            with rasterio.open(sample["sentinel1_desc_ts"]) as f:
                s1_desc = f.read()
            plot_data['sentinel1_desc_ts'] = {
                'plot_type': 'grayscale',
                'data': s1_desc,
                'title': 'Sentinel-1 DESC'
            }
        
        # Plot Sentinel-2 time series if available
        if 'sentinel2_ts' in sample:
            with rasterio.open(sample["sentinel2_ts"]) as f:
                s2_ts = f.read()
            # Rasterio reads as (C, H, W), we need to check number of channels
            # Full Sentinel-2: B02=Blue(1), B03=Green(2), B04=Red(3) (0-indexed)
            num_bands = s2_ts.shape[0]
            if num_bands >= 4:
                rgb_indices = [3, 2, 1]  # R, G, B for full 12-band data
                plot_data['sentinel2_ts'] = {
                    'plot_type': 'rgb',
                    'data': s2_ts,
                    'title': 'Sentinel-2 TS',
                    'rgb_indices': rgb_indices
                }
            elif num_bands == 3:
                rgb_indices = [0, 1, 2]  # Already RGB format
                plot_data['sentinel2_ts'] = {
                    'plot_type': 'rgb',
                    'data': s2_ts,
                    'title': 'Sentinel-2 TS',
                    'rgb_indices': rgb_indices
                }
            elif num_bands == 1:
                # Single band - display as grayscale
                plot_data['sentinel2_ts'] = {
                    'plot_type': 'grayscale',
                    'data': s2_ts,
                    'title': 'Sentinel-2 TS'
                }
            else:
                # Other configurations - use first 3 bands as RGB
                rgb_indices = [0, 1, 2]
                plot_data['sentinel2_ts'] = {
                    'plot_type': 'rgb',
                    'data': s2_ts,
                    'title': 'Sentinel-2 TS',
                    'rgb_indices': rgb_indices
                }
        
        # Plot Sentinel-2 mask if available
        if 'sentinel2_msk_sc' in sample:
            with rasterio.open(sample["sentinel2_msk_sc"]) as f:
                s2_msk = f.read()
            plot_data['sentinel2_msk_sc'] = {
                'plot_type': 'grayscale',
                'data': s2_msk,
                'title': 'Sentinel-2 Mask'
            }
        
        # Create figure with appropriate size
        num_plots = len(plot_data)
        ncols = min(4, num_plots)  # Max 4 columns
        nrows = (num_plots + ncols - 1) // ncols
        
        fig, axs = plt.subplots(
            nrows, ncols, 
            figsize=(ncols * 4, nrows * 4),
            squeeze=False
        )
        axs = axs.flatten()
        
        # Plot each modality
        for idx, (imagery_key, plot_info) in enumerate(plot_data.items()):
            plot_type = plot_info['plot_type']
            data = plot_info['data']
            title = plot_info['title']
            
            if plot_type == 'mask':
                self._plot_mask(data, axs[idx], show_legend=(idx == 0))
            elif plot_type == 'rgb':
                rgb_indices = plot_info['rgb_indices']
                self._plot_rgb_modality(data, axs[idx], title, rgb_indices)
            elif plot_type == 'grayscale':
                self._plot_grayscale_modality(data, axs[idx], title)
            elif plot_type == 'dem':
                self._plot_dem(data, axs[idx], title)
        
        # Hide unused subplots
        for idx in range(num_plots, len(axs)):
            axs[idx].axis('off')
        
        if suptitle:
            fig.suptitle(suptitle, fontsize=16)
        
        plt.tight_layout()
        return fig





class FLAIRHUBToy(FLAIRHUB):
    """
    Toy Version of the FLAIRHUB dataset. For further information refer to the FLAIRHUB dataset.
    """


    download_link: str = (
        'https://storage.gra.cloud.ovh.net/v1/AUTH_366279ce616242ebb14161b7991a8461/defi-ia/flair_hub/FLAIR-HUB_TOY_DATASET.zip'
    )


    def __init__(
        self,
        root: Path = Path('data'),
        transforms: Callable[[dict[str, Tensor]], dict[str, Tensor]] | None = None,
        download: bool = False,
        use_aerial_rgbi: bool = False,
        use_aerial_rlt_pan: bool = False,
        use_spot_rgbi: bool = False,
        use_dem_elev: bool = False,
        use_sentinel1_asc_ts: bool = False,
        use_sentinel1_desc_ts: bool = False,
        use_sentinel2_ts: bool = False,
        use_sentinel2_msk_sc: bool = False,
        dataset_type: Literal["land_cover", "crop_type"] = "land_cover",
    ) -> None:
        """Initialize a new FLAIRHUBToy dataset instance.

        This is a toy/sample version of the FLAIR-HUB dataset intended for testing and
        development purposes. It contains a small subset of the full dataset with all
        available data modalities.

        The toy dataset provides access to 8 complementary data modalities:
        
        * **AERIAL_RGBI**: High-resolution aerial imagery (RGB + NIR, 0.2m)
        * **SPOT_RGBI**: SPOT satellite imagery (RGB + NIR, 1.5m)
        * **DEM_ELEV**: Digital Elevation Model (DSM + DTM, 1m)
        * **AERIAL-RLT_PAN**: Historical aerial panchromatic (1950s)
        * **SENTINEL1-ASC_TS**: Sentinel-1 SAR Ascending time series (VV + VH)
        * **SENTINEL1-DESC_TS**: Sentinel-1 SAR Descending time series (VV + VH)
        * **SENTINEL2_TS**: Sentinel-2 multispectral time series (12 bands, 10m)
        * **SENTINEL2_MSK-SC**: Sentinel-2 scene classification mask

        And two types of labels:
        
        * **land_cover**: 19 classes from COSIA annotation
        * **crop_type**: 23 crop classes from LPIS annotation

        Args:
            root: Root directory where toy dataset can be found or will be downloaded.
            transforms: Optional transforms to apply to samples.
            download: If True, download the toy dataset if not found (~10 MB).
            use_aerial_rgbi: If True, load high-resolution aerial RGBI imagery.
            use_aerial_rlt_pan: If True, load historical aerial panchromatic imagery.
            use_spot_rgbi: If True, load SPOT satellite RGBI imagery.
            use_dem_elev: If True, load Digital Elevation Model data.
            use_sentinel1_asc_ts: If True, load Sentinel-1 Ascending time series.
            use_sentinel1_desc_ts: If True, load Sentinel-1 Descending time series.
            use_sentinel2_ts: If True, load Sentinel-2 multispectral time series.
            use_sentinel2_msk_sc: If True, load Sentinel-2 scene classification masks.
            dataset_type: Type of labels - 'land_cover' (19 classes) or 'crop_type' (23 classes).

        Raises:
            DatasetNotFoundError: If dataset is not found and *download* is False.
            ValueError: If *dataset_type* is not 'land_cover' or 'crop_type'.
            FileNotFoundError: If a requested modality file is missing from the toy dataset.

        Warning:
            This is a TOY DATASET for testing only. Do not use for actual training or
            evaluation. Use the full FLAIRHUB dataset for research purposes.

        Examples:
            >>> # Load toy dataset with aerial imagery
            >>> dataset = FLAIRHUBToy(
            ...     root='data',
            ...     download=True,
            ...     use_aerial_rgbi=True,
            ...     dataset_type='land_cover'
            ... )
            >>> len(dataset)
            42
            >>> 
            >>> # Load all modalities for testing multi-modal pipelines
            >>> dataset = FLAIRHUBToy(
            ...     root='data',
            ...     download=True,
            ...     use_aerial_rgbi=True,
            ...     use_spot_rgbi=True,
            ...     use_dem_elev=True,
            ...     use_sentinel2_ts=True,
            ...     dataset_type='land_cover'
            ... )
            >>> sample = dataset[0]
            >>> print(sample.keys())
            dict_keys(['mask', 'aerial_rgbi', 'spot_rgbi', 'dem_elev', 'sentinel2_ts'])
            >>> 
            >>> # Visualize all loaded modalities
            >>> fig = dataset.plot(sample, suptitle='Multi-modal Sample')

        See Also:
            FLAIRHUB: Full dataset class for production use
        """
        print('-' * 80)
        print('WARNING: Using toy dataset.')
        print('This dataset should be used for testing purposes only.')
        print(
            'Disabling use_toy-flag when initializing the dataset will initialize the full dataset.'
        )
        print('-' * 80)
        if dataset_type not in ["land_cover", "crop_type"]:
            raise ValueError("dataset_type must be either 'land_cover' or 'crop_type'")
        self.root = root
        self.transforms = transforms
        self.download = download
        self.dataset_type = dataset_type
        
        # Store which modalities to use
        self.use_aerial_rgbi = use_aerial_rgbi
        self.use_aerial_rlt_pan = use_aerial_rlt_pan
        self.use_spot_rgbi = use_spot_rgbi
        self.use_dem_elev = use_dem_elev
        self.use_sentinel1_asc_ts = use_sentinel1_asc_ts
        self.use_sentinel1_desc_ts = use_sentinel1_desc_ts
        self.use_sentinel2_ts = use_sentinel2_ts
        self.use_sentinel2_msk_sc = use_sentinel2_msk_sc
        
        super().__init__(
            root=root,
            transforms=transforms,
            download=download,
            checksum=False,
            use_aerial_rgbi=use_aerial_rgbi,
            use_aerial_rlt_pan=use_aerial_rlt_pan,
            use_dem_elev=use_dem_elev,
            use_spot_rgbi=use_spot_rgbi,
            use_sentinel2_ts=use_sentinel2_ts,
            use_sentinel1_asc_ts=use_sentinel1_asc_ts,
            use_sentinel1_desc_ts=use_sentinel1_desc_ts,
            dataset_type=dataset_type,
        )

    def _verify(self) -> None:
        """Verify the integrity of the dataset."""
        toy_dir = Path(self.root) / 'FLAIR-HUB_TOY'
        toy_zip = Path(self.root) / 'FLAIR-HUB_TOY.zip'

        if toy_dir.is_dir():
            print(str(toy_dir))
            print('Toy dataset downloaded and extracted already...')
            return

        if toy_zip.is_file():
            print('Extracting toy dataset...')
            self._extract()
            return

        if not self.download:
            raise DatasetNotFoundError(self)

        self._download()
        self._extract()


        self.files = self._load_files()
    
    def _load_files(self) -> list[dict[str, str]]:
        """Load the files for the toy dataset.
        
        Returns:
            List of dictionaries with paths to each modality for each sample
        """
        files_list = []

        # Determine which label directory to use based on dataset type
        if self.dataset_type == "land_cover":
            label_dir = "AERIAL_LABEL-COSIA"
        elif self.dataset_type == "crop_type":
            label_dir = "ALL_LABEL-LPIS"
        
        filename_glob = f'D*_{label_dir}/*/*.tif'

        # Define mapping of data modalities to their directory names
        # Format: (use_flag_name, key_name, directory_name)
        modalities = [
            ('use_aerial_rgbi', 'aerial_rgbi', 'AERIAL_RGBI'),
            ('use_spot_rgbi', 'spot_rgbi', 'SPOT_RGBI'),
            ('use_dem_elev', 'dem_elev', 'DEM_ELEV'),
            ('use_sentinel1_asc_ts', 'sentinel1_asc_ts', 'SENTINEL1-ASC_TS'),
            ('use_sentinel1_desc_ts', 'sentinel1_desc_ts', 'SENTINEL1-DESC_TS'),
            ('use_sentinel2_ts', 'sentinel2_ts', 'SENTINEL2_TS'),
            ('use_sentinel2_msk_sc', 'sentinel2_msk_sc', 'SENTINEL2_MSK-SC'),
        ]

        # Iterate through all label files and build file dictionaries
        for label_path in (Path(self.root) / "FLAIR-HUB_TOY").glob(filename_glob):
            file_dict = {}
            file_dict["mask"] = label_path
            
            # Add each requested modality
            for use_flag, key_name, dir_name in modalities:
                if getattr(self, use_flag):
                    file_path = Path(str(label_path).replace(label_dir, dir_name))
                    if not file_path.exists():
                        raise FileNotFoundError(
                            f"{dir_name} file not found: {file_path}\n"
                            f"Expected to exist for label: {label_path}"
                        )
                    file_dict[key_name] = file_path
            
            # Handle historical aerial data (has different year pattern)
            if self.use_aerial_rlt_pan:
                # Extract the department-year part (e.g., "D006-2020")
                label_str = str(label_path)
                dept_year = label_str.split('_')[0].split('/')[-1]  # e.g., "D006-2020"
                dept = dept_year.split('-')[0]  # e.g., "D006"
                
                # Replace year with "195X" for historical data
                aerial_rlt_path = label_str.replace(
                    f"{dept_year}_{label_dir}", 
                    f"{dept}-195X_AERIAL-RLT_PAN"
                )
                aerial_rlt_path = Path(aerial_rlt_path)
                
                if not aerial_rlt_path.exists():
                    raise FileNotFoundError(
                        f"AERIAL-RLT_PAN file not found: {aerial_rlt_path}\n"
                        f"Expected to exist for label: {label_path}"
                    )
                file_dict['aerial_rlt_pan'] = aerial_rlt_path
            
            files_list.append(file_dict)

        return files_list
    
    
    def _download(self) -> None:
        """Download the dataset."""
        download_url(
            self.download_link,
            self.root,
        )

    def _extract(self) -> None:
        """Extract the dataset."""
        zipfile = Path(self.root) / self.download_link.split('/')[-1]
        assert zipfile.is_file()
        extract_archive(zipfile)
        zipfile.unlink()
        print(f"Toy dataset extracted and deleted.")
