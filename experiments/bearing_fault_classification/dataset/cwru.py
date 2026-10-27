from scipy.io import loadmat
import numpy as np
import os, re
import errno
import urllib.request as urllib
from scipy.io import loadmat
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
import torch

# https://engineering.case.edu/bearingdatacenter/apparatus-and-procedures
faults_idx = {
    "Normal": 0,
    "0.007-Ball": 1,
    "0.014-Ball": 2,
    "0.021-Ball": 3,
    "0.007-InnerRace": 4,
    "0.014-InnerRace": 5,
    "0.021-InnerRace": 6,
    "0.007-OuterRace3": 7,
    "0.007-OuterRace6": 7,
    "0.007-OuterRace12": 7,
    "0.014-OuterRace3": 8,
    "0.014-OuterRace12": 8,
    "0.014-OuterRace6": 8,
    "0.021-OuterRace6": 9,
    "0.021-OuterRace3": 9,
    "0.021-OuterRace12": 9,
}

FOLDER = "cwru-dataset"


def filter_key(keys):
    fkeys = []
    for key in keys:
        matchObj = re.match(r"(.*)FE_time", key, re.M | re.I)
        if matchObj:
            fkeys.append(matchObj.group(1))
    return fkeys[0] + "DE_time", fkeys[0] + "FE_time"


def get_class(exp, fault):
    if fault == "Normal":
        return 0
    return faults_idx[fault]


class CrwuDataset():
    def __init__(
        self,
        rdir="data/CWRU/",
        rpms=["1797", "1772", "1750", "1730"],
        fault_location="DriveEnd",
        seq_len=2048,
    ):
        super().__init__()
        self.rdir = rdir
        self.rpms = rpms
        self.fault_location = fault_location
        self.seq_len = seq_len

        if fault_location not in ["DriveEnd", "FanEnd"]:
            raise ValueError("Fault location must be either DriveEnd or FanEnd")

        for rpm in rpms:
            if rpm not in ["1797", "1772", "1750", "1730"]:
                raise ValueError("RPM must be one of 1797, 1772, 1750, 1730")

        if fault_location == "DriveEnd":
            self.signal_key = "DE"
            self.set = "12DriveEndFault"
        else:
            self.signal_key = "FE"
            self.set = "12FanEndFault"

        fmeta = os.path.join(os.path.dirname(os.path.abspath(__file__)), "metadata.txt")
        with open(fmeta, encoding="utf-8") as metadata_file:
            all_lines = metadata_file.readlines()
        infos = []
        for line in all_lines:
            l = line.split()
            if (l[0] == self.set or l[0] == "NormalBaseline") and l[1] in rpms:
                if (
                    "Normal" in l[2]
                    or "0.007" in l[2]
                    or "0.014" in l[2]
                    or "0.021" in l[2]
                ):
                    if faults_idx.get(l[2], -1) != -1:
                        infos.append(l)

        infos = sorted(infos, key=lambda line: get_class(line[0], line[2]))

        self._load_data(rdir, infos)
        self.shuffle()

    def window_size(self):
        return self.seq_len

    def _mkdir(self, path):
        try:
            os.makedirs(path)
        except OSError as exc:
            if exc.errno == errno.EEXIST and os.path.isdir(path):
                pass
            else:
                print("can't create directory '{}''".format(path))
                exit(1)

    def _download(self, fpath, link, retryCount=1):
        print(link + " Downloading to: '{}'".format(fpath))
        try:
            urllib.urlretrieve(link, fpath)
        except:
            if retryCount > 0:
                os.remove(fpath)
                self._download(fpath, link, retryCount - 1)
            else:
                print("can't download file '{}'".format(link))
                exit(1)

    def _load_data(self, rdir, infos):
        if self._load_preprocessed():
            return

        X = np.zeros((0, self.seq_len, 1))
        y = []

        for info in infos:
            # Directory of this file
            fdir = os.path.join(rdir, info[0], info[1])
            self._mkdir(fdir)
            fpath = os.path.join(fdir, info[2] + ".mat")

            if not os.path.exists(fpath):
                self._download(fpath, info[3].rstrip("\n"), 10)

            mat_dict = loadmat(fpath)
            driveEndKey, fanEndKey = filter_key(mat_dict.keys())
            time_series = None
            if self.signal_key == "DE":
                time_series = mat_dict[driveEndKey]
            else:
                time_series = mat_dict[fanEndKey]

            n_samples = (
                len(time_series) - (len(time_series) % self.seq_len)
            ) / self.seq_len

            # Process training data
            clips = np.zeros((0, 1))
            for cut in shuffle(list(range(int(n_samples)))):
                clips = np.vstack(
                    (
                        clips,
                        time_series[cut * self.seq_len : (cut + 1) * self.seq_len],
                    )
                )
            clips = clips.reshape(-1, self.seq_len, 1)
            X = np.vstack((X, clips))
            y.extend([get_class(info[0], info[2]) for _ in range(len(clips))])

        X = X.reshape(-1, 1, self.seq_len)

        self.X = X
        self.y = np.array(y)

        self.presist()

    def ensure_dir(self, path):
        if not os.path.exists(path):
            os.makedirs(path)

    def _load_preprocessed(self):
        x_path, y_path = self._cache_paths()
        if not os.path.exists(x_path) or not os.path.exists(y_path):
            return False
        print("Loading preprocessed data...")
        self.X = np.load(x_path)
        self.y = np.load(y_path)
        return True

    def presist(self):
        self.ensure_dir(os.path.join(self.rdir, FOLDER))
        x_path, y_path = self._cache_paths()
        np.save(x_path, self.X)
        np.save(y_path, self.y)

    def _cache_paths(self):
        rpm_tag = "-".join(self.rpms)
        prefix = f"{self.signal_key}_{rpm_tag}_{self.seq_len}"
        cache_dir = os.path.join(self.rdir, FOLDER)
        return (
            os.path.join(cache_dir, f"X_{prefix}.npy"),
            os.path.join(cache_dir, f"y_{prefix}.npy"),
        )

    def shuffle(self):
        indices = np.arange(len(self.X))
        np.random.shuffle(indices)
        self.X = self.X[indices]
        self.y = np.array(tuple(self.y[i] for i in indices))

    def __len__(self):
        return len(self.X)

    def __getitem__(self, index):
        # Store each sample as [time, features] for the classifier.
        signal = torch.tensor(self.X[index].T).float()
        return signal, torch.tensor(self.y[index]).long()

    def set_inputs(self, X: torch.Tensor):
        self.X = X

    def set_targets(self, y: torch.Tensor):
        self.y = y

    def stack(self, X: torch.Tensor, y: torch.Tensor):
        self.X = np.vstack((self.X, X))
        self.y = np.hstack((self.y, y))

    def inputs(self):
        return self.X

    def targets(self):
        return self.y

    def labels(self):
        return [
            "Normal",
            "0.007-Ball",
            "0.014-Ball",
            "0.021-Ball",
            "0.007-InnerRace",
            "0.014-InnerRace",
            "0.021-InnerRace",
            "0.007-OuterRace",
            "0.014-OuterRace",
            "0.021-OuterRace",
        ]