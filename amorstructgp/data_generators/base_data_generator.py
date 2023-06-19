# Copyright (c) 2023 Robert Bosch GmbH
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published
# by the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

# Author: Matthias Bitzer, matthias.bitzer3@de.bosch.com
from abc import ABC, abstractmethod
from typing import Optional, Tuple


from amorstructgp.data_generators.simulator import SimulatedDataset


class BaseDataGenerator(ABC):
    @staticmethod
    def get_main_file_names() -> Tuple[str, str, str]:
        """
        Method that should return the file names of files that contain the uuid lists
        for the train, val and test set

        Returns:
            str : file name of the file that contains uuid list of training set
            str : file name of the file that contains uuid list of validation set
            str : file name of the file that contains uuid list of test set
        """
        raise NotImplementedError

    @abstractmethod
    def generate_simulated_data(self, n_train: int, n_val: int, n_test: int, folder: str):
        """
        Main function that generates a dataset of datasets - and splits it in train val and test set
        It should generate SimulatedDataset objects and store them together with files that contain lists
        of the uuids of the SimulatedDataset objects that are associated with one of the three sets

        Arguments:
         n_train : int number of training datasets
         n_val: int number of validation datasets
         n_test: int number of test datasets
         folder: str - base folder where datasets and managment files should be stored
        """
        raise NotImplementedError

    @abstractmethod
    def generate_one_sample() -> SimulatedDataset:
        """
        Generates one SimulateDataset object and returns it - is used in Dataset objects that
        generates new random SimulatedDatasets in each batch rather than loading them from files
        """
        raise NotImplementedError

    @abstractmethod
    def get_max_dimension(self, D: Optional[int] = None) -> int:
        """
        Returns the maximum dimenion it will generate
        """
        raise NotImplementedError

    @staticmethod
    def get_dimension_index_file_names() -> Tuple[str, str, str]:
        """
        Method that should return the file names of files that contain the json indexes that map input dimensions to the list of uuids
        associated with datasets with that input dimension respectively for the train, val and test set

        Returns:
            str : file name of the file that contains json index (input dimension -> uuid list) of training set
            str : file name of the file that contains json index (input dimension -> uuid list) of validation set
            str : file name of the file that contains json index (input dimension -> uuid list) of test set
        """
        raise NotImplementedError

    @abstractmethod
    def set_initial_seed(self, seed: int):
        """
        Method for setting a seed

        Arguments:
            int  : seed value
        """
        raise NotImplementedError
