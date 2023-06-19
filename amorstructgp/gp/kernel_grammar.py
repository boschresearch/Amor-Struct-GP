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
import random
from typing import Dict, List, Optional, Tuple, Union
from enum import Enum
from abc import ABC, abstractmethod
import numpy as np
import pickle
import logging
from copy import deepcopy
from amorstructgp.gp.gpytorch_kernels.elementary_kernels_pytorch import BaseElementaryKernelPytorch

logger = logging.getLogger(__name__)


class KernelGrammarOperator(Enum):
    ADD = 1
    MULTIPLY = 2


class BaseKernelsLibrary(Enum):
    GPYTORCH = 1


ElementaryKernel = BaseElementaryKernelPytorch


class KernelGrammarOperatorWrapper:
    """
    A way to wrap an operator enum into an object to add additinal inforamation. Mainly used for the CP operator
    as it needs the dimension at which it is applied. The wrapper acts as an KernelGrammarOperator object in the
    sense that comparisons can be made against KernelGrammarOperator objects.
    """

    def __init__(self, operator: KernelGrammarOperator) -> None:
        assert isinstance(operator, KernelGrammarOperator)
        self.operator = operator
        self.dimension = None

    def set_dimension(self, dimension: int):
        self.dimension = dimension

    def get_dimension(self):
        return self.dimension

    def __eq__(self, operator: object) -> bool:
        if isinstance(operator, KernelGrammarOperator):
            return self.operator == operator
        elif isinstance(operator, self.__class__):
            return self.operator == operator.operator

    def __ne__(self, operator: object) -> bool:
        return not self.__eq__(operator)


class BaseKernelGrammarExpression(ABC):
    """
    Base Object that represents a symbolic expression that corresponds to a kernel. It either represents a basic kernel (ElementaryKernelGrammarExpression)
    or a binary tree consisting of operator suchs as ADD and MULTIPLY as its nodes and base kernels as its children (KernelGrammarExpression). An instance of this class
    always correconds to a gpytorch.kernels.Kernel instance which can be resolved via the get_kernel() method.
    """

    @abstractmethod
    def get_kernel(self) -> ElementaryKernel:
        """
        Method to resolve/get the corresponding kernel object
        Return
            gpytorch.kernels.Kernel - returns the associated kernel object
        """
        raise NotImplementedError

    @abstractmethod
    def deep_copy(self):
        """
        Returns a deep copy of itself - creates copies of all base kernels and subexpressions
        - thus if weight sharing was active between kernels in subexpression this is not the case for the returned object
        Return:
         BaseKernelGrammarExpression - a deep copy of its self
        """
        raise NotImplementedError

    @abstractmethod
    def count_elementary_expressions(self) -> int:
        """
        Counts number of elementary expressions (instances of ElementaryKernelGrammarExpression) inside the expression
        """
        raise NotImplementedError

    @abstractmethod
    def count_operators(self):
        """
        Counts the number of operators in the binary tree (only greater than 0 for instances of KernelGrammarExpression)
        """
        raise NotImplementedError

    @abstractmethod
    def get_name(self):
        """
        Returns a unique name based on the tree structure (operators) and the base kernels in the leaves
        """
        raise NotImplementedError

    @abstractmethod
    def get_hash(self) -> Tuple[int, List[Tuple[int, int, int]]]:
        """
        Calcuates hash of the expression based on hashes of all containing subexpressions -> makes it possible to count number
        of identical subexpressions - is used by the get_subtree_dict method
        Return:
         int - hash value of expression - hash is generated recursivly and is invariant to rotation of the nodes in the binary tree
         List[(int,int,int)] - list over all subexpressions with tuples containing: (hash values of subexpressions,number of elementary expressions inside subexpression,depth of the root of the subexpression)
        """
        raise NotImplementedError

    @abstractmethod
    def get_subtree_dict(self) -> Dict[int, List]:
        """
        Generates dict where each key is a hash of a subexpression and the value is a List containing the number how often the subexpression appeared inside this expression at index 0 -
        this method allows for example the tree grammar kernel kernel to count how often the same subexpression appeared in two different expressions - at index 1 a SubtreeMetaInformation
        object is stored associated to the subexpression
        """
        raise NotImplementedError

    @abstractmethod
    def get_input_dimension(self) -> int:
        raise NotImplementedError

    @abstractmethod
    def get_operator(self) -> Optional[KernelGrammarOperator]:
        raise NotImplementedError

    @abstractmethod
    def get_indexes_of_subexpression(self) -> List[List[int]]:
        """Returns lists of the form [0,0,1,0,...] for each subexpression
        which indexes each subexpression with its own list -> it specifies the path in the tree to the subexpression
        [-1] is a special index list referencing to self/root - expressions corresponding to the index_list can be retrieved by
        get_expression_at_index
        """
        raise NotImplementedError

    @abstractmethod
    def get_indexes_of_elementary_expressions(self) -> List[List[int]]:
        raise NotImplementedError

    @abstractmethod
    def get_expression_at_index(self, index_list: List[int]):
        """
        Returns subexpression at index_list where index_list has the form [0,1,1,...] specifying the way down the tree
        0 go down expression1, 1 go down expression2. [-1] is the index_list for the expression itself
        """
        raise NotImplementedError

    @abstractmethod
    def get_base_kernel_library(self) -> BaseKernelsLibrary:
        raise NotImplementedError

    def __str__(self) -> str:
        return self.get_name()


class SubtreeMetaInformation:
    def __init__(self, hash_value: int, num_elementary: int, depth_counter: int):
        self.hash_value = hash_value
        self.num_elementary = num_elementary
        self.depth_counter = depth_counter
        self.upstream_operators = []

    def increase_depth_count(self):
        self.depth_counter += 1

    def add_upstream_operator(self, operator: KernelGrammarOperator):
        self.upstream_operators.append(operator)

    def __str__(self):
        return (
            "[Hash: "
            + str(self.hash_value)
            + " num_elementary: "
            + str(self.num_elementary)
            + " depth_counter: "
            + str(self.depth_counter)
            + " upstream_operators: "
            + str(self.upstream_operators)
            + "]"
        )


class ElementarySubtreeMetaInformation(SubtreeMetaInformation):
    def __init__(self, hash_value: int, num_elementary: int, depth_counter: int, elementary_name: str):
        super().__init__(hash_value, num_elementary, depth_counter)
        self.elementary_name = elementary_name


class ElementaryPathMetaInformation:
    def __init__(self, elementary_hash: int, upstream_operator_list: List[KernelGrammarOperator]):
        self.elementary_hash = elementary_hash
        self.upstream_operator_list = upstream_operator_list
        self.hash = hash(str(self.upstream_operator_list + [self.elementary_hash]))

    def generate_operator_count_dict(self):
        operator_count_dict = {}
        for operator in self.upstream_operator_list:
            if operator in operator_count_dict:
                operator_count_dict[operator] += 1
            else:
                operator_count_dict[operator] = 1
        return operator_count_dict

    def get_hash(self):
        return self.hash


class ElementaryKernelGrammarExpression(BaseKernelGrammarExpression):
    def __init__(self, kernel: ElementaryKernel):
        self.check_kernel_validity(kernel)
        self.kernel = kernel
        self.generator_name = None
        self.base_kernel_library = self.extract_kernel_library(kernel)

    def set_kernel(self, kernel: ElementaryKernel):
        self.check_kernel_validity(kernel)
        kernel_library = self.extract_kernel_library(kernel)
        assert self.base_kernel_library == kernel_library, "Kernel library can not be changed after object creation"
        self.kernel = kernel

    def check_kernel_validity(self, kernel: ElementaryKernel):
        assert isinstance(kernel, BaseElementaryKernelPytorch)

    def extract_kernel_library(self, kernel: ElementaryKernel):
        if isinstance(kernel, BaseElementaryKernelPytorch):
            return BaseKernelsLibrary.GPYTORCH
        else:
            assert False

    def deep_copy(self):
        if isinstance(self.kernel, BaseElementaryKernelPytorch):
            new_kernel = deepcopy(self.kernel)
            assert isinstance(new_kernel, BaseElementaryKernelPytorch)
            return ElementaryKernelGrammarExpression(new_kernel)

    def count_elementary_expressions(self):
        return 1

    def count_operators(self):
        return 0

    def get_input_dimension(self):
        return self.kernel.get_input_dimension()

    def get_name(self):
        return self.kernel.name

    def get_operator(self) -> Optional[KernelGrammarOperator]:
        return None

    def get_hash(self):
        hash_value = hash(self.kernel.name)
        subtree_meta_info = ElementarySubtreeMetaInformation(hash_value, 1, 0, self.get_name())
        return hash_value, [subtree_meta_info]

    def get_kernel(self):
        return self.kernel

    def get_indexes_of_subexpression_internal(self) -> List[List[int]]:
        return [[None]]

    def get_indexes_of_subexpression(self) -> List[List[int]]:
        return [[-1]]

    def get_indexes_of_elementary_expressions(self) -> List[List[int]]:
        return [[-1]]

    def get_expression_at_index(self, index_list) -> BaseKernelGrammarExpression:
        assert len(index_list) == 1
        assert index_list[0] == -1
        return self

    def get_subtree_dict(self):
        subtree_meta_info = self.get_hash()[1][0]
        subtree_dict = {}
        subtree_dict[self.get_hash()[0]] = [1, subtree_meta_info]
        return subtree_dict

    def get_base_kernel_library(self) -> BaseKernelsLibrary:
        return self.base_kernel_library


class KernelGrammarExpression(BaseKernelGrammarExpression):
    def __init__(self, expression1: BaseKernelGrammarExpression, expression2: BaseKernelGrammarExpression, operator: KernelGrammarOperator):
        self.expression1: BaseKernelGrammarExpression = expression1
        self.expression2: BaseKernelGrammarExpression = expression2
        assert (
            self.expression1.get_base_kernel_library() == self.expression2.get_base_kernel_library()
        ), "Fusion of kernels of different base libraries not possible"
        self.base_kernel_library = self.expression1.get_base_kernel_library()
        self.operator: KernelGrammarOperator = operator
        self.generator_name = None

    def get_kernel(self):
        if self.operator == KernelGrammarOperator.ADD:
            return self.expression1.get_kernel() + self.expression2.get_kernel()
        elif self.operator == KernelGrammarOperator.MULTIPLY:
            return self.expression1.get_kernel() * self.expression2.get_kernel()

    def get_operator(self) -> Optional[KernelGrammarOperator]:
        return self.operator

    def deep_copy(self):
        new_expression_1 = self.expression1.deep_copy()
        new_expression_2 = self.expression2.deep_copy()
        new_expression = KernelGrammarExpression(expression1=new_expression_1, expression2=new_expression_2, operator=self.operator)
        return new_expression

    def get_left_expression(self):
        return self.expression1

    def get_right_expression(self):
        return self.expression2

    def get_input_dimension(self):
        assert self.expression1.get_input_dimension() == self.expression2.get_input_dimension()
        return self.expression1.get_input_dimension()

    def count_elementary_expressions(self):
        return self.expression1.count_elementary_expressions() + self.expression2.count_elementary_expressions()

    def count_operators(self):
        return self.expression1.count_operators() + self.expression2.count_operators() + 1

    def get_base_kernel_library(self) -> BaseKernelsLibrary:
        return self.base_kernel_library

    def get_name(self):
        if self.operator == KernelGrammarOperator.ADD:
            return "(" + self.expression1.get_name() + " ADD " + self.expression2.get_name() + ")"
        elif self.operator == KernelGrammarOperator.MULTIPLY:
            return "(" + self.expression1.get_name() + " MULTIPLY " + self.expression2.get_name() + ")"

    def get_indexes_of_subexpression_internal(self) -> List[List[int]]:
        list_left = [[0]]
        list_right = [[1]]
        list_expression_1 = self.expression1.get_indexes_of_subexpression_internal()
        for index_list in list_expression_1:
            extended_index_list = [0] + index_list
            list_left.append(extended_index_list)
        list_expression_2 = self.expression2.get_indexes_of_subexpression_internal()
        for index_list in list_expression_2:
            extended_index_list = [1] + index_list
            list_right.append(extended_index_list)
        return list_left + list_right

    def get_indexes_of_subexpression(self) -> List[List[int]]:
        complete_list = self.get_indexes_of_subexpression_internal()
        reduced_list = [[-1]]
        for index_list in complete_list:
            if index_list[-1] is not None:
                reduced_list.append(index_list)
        return reduced_list

    def get_indexes_of_elementary_expressions(self) -> List[List[int]]:
        complete_list = self.get_indexes_of_subexpression_internal()
        reduced_list = []
        for index_list in complete_list:
            if index_list[-1] is None:
                reduced_list.append(index_list[:-1])
        return reduced_list

    def get_expression_at_index(self, index_list: List[int]) -> BaseKernelGrammarExpression:
        if len(index_list) == 1:
            if index_list[0] == 0:
                return self.expression1
            elif index_list[0] == 1:
                return self.expression2
            elif index_list[0] == -1:
                return self
        else:
            if index_list[0] == 0:
                assert isinstance(self.expression1, KernelGrammarExpression)
                return self.expression1.get_expression_at_index(index_list[1:])
            elif index_list[0] == 1:
                assert isinstance(self.expression2, KernelGrammarExpression)
                return self.expression2.get_expression_at_index(index_list[1:])

    def set_expression_at_index(self, index_list: List[int], expression: BaseKernelGrammarExpression):
        """
        Exchanges subexpression at index_list with the expression in the argument where index_list has the form [0,1,1,...] specifying the way down the tree
        0 go down expression1, 1 go down expression2. Changing of itself (index list [-1]) is not allowed
        """
        if len(index_list) == 1:
            if index_list[0] == 0:
                self.expression1 = expression
            elif index_list[0] == 1:
                self.expression2 = expression
            elif index_list[0] == -1:
                assert False
        else:
            if index_list[0] == 0:
                assert isinstance(self.expression1, KernelGrammarExpression)
                return self.expression1.set_expression_at_index(index_list[1:], expression)
            elif index_list[0] == 1:
                assert isinstance(self.expression2, KernelGrammarExpression)
                return self.expression2.set_expression_at_index(index_list[1:], expression)

    def get_operator_hash(self):
        if self.operator == KernelGrammarOperator.ADD:
            hash_value = hash("ADD")
        elif self.operator == KernelGrammarOperator.MULTIPLY:
            hash_value = hash("MULTIPLY")
        return hash_value

    def get_hash(self):
        hash_collection = []
        hash_collection.append(self.get_operator_hash())
        hash_expr1, meta_info_list_expr1 = self.expression1.get_hash()
        hash_expr2, meta_info_list_expr2 = self.expression2.get_hash()
        for element in meta_info_list_expr1:
            element.increase_depth_count()
            element.add_upstream_operator(self.operator)
        for element in meta_info_list_expr2:
            element.increase_depth_count()
            element.add_upstream_operator(self.operator)
        hash_collection.append(hash_expr1)
        hash_collection.append(hash_expr2)
        hash_collection.sort()
        hash_value = hash(str(hash_collection))
        num_elementary_expressions = self.count_elementary_expressions()
        subtree_meta_info = SubtreeMetaInformation(hash_value, num_elementary_expressions, 0)
        subtree_meta_info_list = [subtree_meta_info] + meta_info_list_expr1 + meta_info_list_expr2
        return hash_value, subtree_meta_info_list

    def get_subtree_dict(self):
        _, subtree_meta_info_list = self.get_hash()
        subtree_dict = {}
        for subtree_meta_info in subtree_meta_info_list:
            hash_value = subtree_meta_info.hash_value
            # num_elementary = subtree_meta_info.num_elementary
            if hash_value in subtree_dict:
                subtree_dict[hash_value][0] += 1
            else:
                subtree_dict[hash_value] = [1, subtree_meta_info]

        return subtree_dict


class KernelGrammarExpressionTransformer:
    @staticmethod
    def multiply_expressions(expression_list: List[BaseKernelGrammarExpression], make_deep_copy=True) -> BaseKernelGrammarExpression:
        return KernelGrammarExpressionTransformer.combine_all_expressions_via_operator(
            KernelGrammarOperator.MULTIPLY, expression_list, make_deep_copy
        )

    @staticmethod
    def add_expressions(expression_list: List[BaseKernelGrammarExpression], make_deep_copy=True) -> BaseKernelGrammarExpression:
        return KernelGrammarExpressionTransformer.combine_all_expressions_via_operator(
            KernelGrammarOperator.ADD, expression_list, make_deep_copy
        )

    @staticmethod
    def combine_all_expressions_via_operator(
        operator: KernelGrammarOperator, expression_list: List[BaseKernelGrammarExpression], make_deep_copy=True
    ) -> BaseKernelGrammarExpression:
        if len(expression_list) == 1:
            return expression_list[0]
        assert len(expression_list) > 1
        new_expression = expression_list[0]
        for i in range(1, len(expression_list)):
            if i % 2 == 0:
                new_expression = KernelGrammarExpression(new_expression, expression_list[i], operator=operator)
            else:
                new_expression = KernelGrammarExpression(expression_list[i], new_expression, operator=operator)
        if make_deep_copy:
            return new_expression.deep_copy()
        else:
            return new_expression


if __name__ == "__main__":
    pass
