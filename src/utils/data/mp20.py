import os
import urllib.request

from .csv_dataset import CSVDataset
from src.utils.download_bar import DownloadProgressBar

url_mp20 = "https://raw.githubusercontent.com/txie-93/cdvae/main/data/mp_20/"


class MP20(CSVDataset):
    def __init__(
        self,
        root: str,
        subset: str,
        transform=None,
        pre_filter=None,
        warn: bool = False,
        multithread: bool = True,
        verbose: bool = True,
    ):
        assert subset in ("train", "val", "test")

        self.subset = subset

        super().__init__(
            root,
            transform=transform,
            pre_filter=pre_filter,
            warn=warn,
            multithread=multithread,
            verbose=verbose,
        )

    @property
    def raw_file_names(self):
        return [f"{self.subset}.csv"]

    @property
    def processed_file_names(self):
        return [f"{self.subset}.hdf5"]

    def download(self):
        url = os.path.join(url_mp20, self.raw_file_names[0])

        with DownloadProgressBar(
            unit="B",
            unit_scale=True,
            miniters=1,
            desc=f"downloading {self.raw_file_names[0]}",
        ) as t:
            urllib.request.urlretrieve(
                url,
                filename=os.path.join(self.raw_dir, self.raw_file_names[0]),
                reporthook=t.update_to,
            )

    def load(self):
        processed_file = os.path.join(self.processed_dir, self.processed_file_names[0])

        self.load_hdf5(processed_file)

    def process(self):
        raw_file = os.path.join(self.raw_dir, self.raw_file_names[0])
        processed_file = os.path.join(self.processed_dir, self.processed_file_names[0])

        self.process_csv(
            raw_file,
            processed_file,
            loading_description=f"proprocess {self.subset} set of MP-20",
        )
