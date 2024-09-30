# SPDX-License-Identifier: MIT
# This file implements the SCOPE algorithm for log parsing.

from abc import ABC, abstractmethod
from typing import cast, Collection, IO, Iterable, MutableMapping, MutableSequence, Optional, Sequence, Tuple, \
    TYPE_CHECKING, TypeVar, Union
from enum import Enum
from cachetools import LRUCache, Cache
from collections import defaultdict

from simple_profiler import Profiler, NullProfiler


class ClusterType(Enum):
    TEMPLATE = 1
    LOG = 2

class Cluster(ABC):
    def __init__(self, cluster_id: int = 0, tokens: Iterable[str] = "", cluster_type: ClusterType = ClusterType.TEMPLATE) -> None:
      self.cluster_id = cluster_id
      self.size = 0
      self.cluster_type = cluster_type
      self.tokens = tuple(tokens)

    def get_tokens(self) -> Tuple[str]:
        return self.tokens

    def set_tokens(self, tokens) -> None:
        self.tokens = tuple(tokens)

    def __str__(self) -> str:
        return f"ID={str(self.cluster_id).ljust(5)} : size={str(self.size).ljust(10)}: {self.get_template()}"

    @abstractmethod
    def get_template(self) -> str:
        ...

class LogCluster(Cluster):
    def __init__(self, log_tokens: Iterable[str] = "This is a default log message", cluster_id: int = 0) -> None:
      super().__init__(cluster_id, log_tokens, ClusterType.LOG)
      self.log_tokens = tuple(log_tokens)

    def get_template(self) -> str:
        return ' '.join(self.log_tokens)

    def set_template(self, template: Iterable[str]) -> None:
        self.log_tokens = tuple(template)
        self.set_tokens(template)

class TemplateCluster(Cluster):
    def __init__(self, template_tokens: Iterable[str] = "This is a default log template", cluster_id: int = 0, log_cluster_id_pair: tuple =()) -> None:
      super().__init__(cluster_id, template_tokens, ClusterType.TEMPLATE)
      self.template_tokens = tuple(template_tokens)
      self.log_cluster_id_pair = log_cluster_id_pair

    def get_template(self) -> str:
        return ' '.join(self.template_tokens)

    def set_template(self, template: Iterable[str]) -> None:
        self.template_tokens = tuple(template)
        self.set_tokens(template)


_T = TypeVar("_T")
if TYPE_CHECKING:
    class _LRUCache(LRUCache[int, Optional[LogCluster]]):
        #  see https://github.com/python/mypy/issues/4148 for this hack
        ...
else:
    _LRUCache = LRUCache

class LogClusterCache(_LRUCache):
    """
    Least Recently Used (LRU) cache which allows callers to conditionally skip
    cache eviction algorithm when accessing elements.
    """

    def __missing__(self, key: int) -> None:
        return None

    def get(self, key: int, _: Union[Optional[LogCluster], _T] = None) -> Optional[LogCluster]:
        """
        Returns the value of the item with the specified key without updating
        the cache eviction algorithm.
        """
        return Cache.__getitem__(self, key)

class NodeType(Enum):
    ROOT = 1
    INTERMEDIATE = 2
    LEAF = 3

class Node():
    __slots__ = ["node_type", "key_to_child_node", "cluster_ids", "log_message_ids"]

    def __init__(self, node_type: NodeType) -> None:
        self.node_type: NodeType = node_type
        self.key_to_child_node: MutableMapping[str, Node] = {}
        self.cluster_ids: Sequence[int] = []
        self.log_message_ids: Sequence[int] = []


class ScopeBase(ABC):
    def __init__(self,
                 depth: int = 4,
                 sim_th: float = 0.4,
                 max_children: int = 100,
                 max_clusters: Optional[int] = None,
                 extra_delimiters: Sequence[str] = (),
                 profiler: Profiler = NullProfiler(),
                 param_str: str = "<*>",
                 parametrize_numeric_tokens: bool = True) -> None:
        """
        Create a new Scope instance.

        :param depth: max depth levels of log clusters. Minimum is 3.
            For example, for depth==4, Root is considered depth level 1.
            Token count is considered depth level 2.
            First log token is considered depth level 3.
            Log clusters below first token node are considered depth level 4.
        :param sim_th: similarity threshold - if percentage of similar tokens for a log message is below this
            number, a new log cluster will be created.
        :param max_children: max number of children of an internal node
        :param max_clusters: max number of tracked clusters (unlimited by default).
            When this number is reached, model starts replacing old clusters
            with a new ones according to the LRU policy.
        :param extra_delimiters: delimiters to apply when splitting log message into words (in addition to whitespace).
        :param parametrize_numeric_tokens: whether to treat tokens that contains at least one digit
            as template parameters.
        """
        if depth < 3:
            raise ValueError("depth argument must be at least 3")

        self.log_cluster_depth = depth
        self.max_node_depth = depth - 2  # max depth of a prefix tree node, starting from zero
        self.sim_th = sim_th
        self.max_children = max_children
        self.root_node = Node(NodeType.ROOT)
        self.profiler = profiler
        self.extra_delimiters = extra_delimiters
        self.max_clusters = max_clusters
        self.param_str = param_str
        self.parametrize_numeric_tokens = parametrize_numeric_tokens

        self.id_to_fused_templateCluster: MutableMapping[int, Optional[TemplateCluster]] = {}
        self.fused_templateClusters_cnt = 0
        self.id_to_tree_templateCluster: MutableMapping[int, Optional[TemplateCluster]] = {}
        self.tree_templateClusters_cnt = 0
        self.id_to_logCluster: MutableMapping[int, Optional[LogCluster]] = {} #{Log_id: LogCluster, id: log, ...}
        self.logCluster_counter = 0
        self.comparable_logClusters = {} # {len:{log_id: logCluster, id: logCluster, ...}, ..., len:{}, ...}
        self.logClstId_to_templateClstId = defaultdict(dict) # {len:{log_id: template_id, ...}, len:{}, ...}


    @property
    def clusters(self) -> Collection[LogCluster]:
        return cast(Collection[LogCluster], self.id_to_fused_templateCluster.values())

    @staticmethod
    def has_numbers(s: Iterable[str]) -> bool:
        return any(char.isdigit() for char in s)

    def fast_match(self,
                   cluster_ids: Collection[int],
                   tokens: Sequence[str],
                   sim_th: float,
                   include_params: bool,
                   cluster_type: ClusterType = ClusterType.TEMPLATE) -> Optional[LogCluster]:
        """
        Find the best match for a log message (represented as tokens) versus a list of clusters
        :param cluster_ids: List of clusters to match against (represented by their IDs)
        :param tokens: the log message, separated to tokens.
        :param sim_th: minimum required similarity threshold (None will be returned in no clusters reached it)
        :param include_params: consider tokens matched to wildcard parameters in similarity threshold.
        :return: Best match cluster or None
        """
        match_cluster = None

        max_sim: Union[int, float] = -1
        max_param_count = -1
        max_cluster = None

        if cluster_type == ClusterType.TEMPLATE:
            clusters = self.id_to_tree_templateCluster
        else:
            clusters = self.id_to_logCluster

        #print("cluster_ids:", cluster_ids)

        for cluster_id in cluster_ids:
            # Try to retrieve cluster from cache with bypassing eviction
            # algorithm as we are only testing candidates for a match.
            cluster = clusters.get(cluster_id)
            if cluster is None:
                continue
            cur_sim, param_count = self.get_seq_distance(cluster.get_tokens(), tokens, include_params)
            #print("fast_match: tree template:", cluster.get_tokens())
            #print("fast_match: input tokens:", tokens)
            if cur_sim > max_sim or (cur_sim == max_sim and param_count > max_param_count):
                max_sim = cur_sim
                max_param_count = param_count
                max_cluster = cluster

        if max_sim >= sim_th:
            match_cluster = max_cluster
        return match_cluster

    def print_tree(self, file: Optional[IO[str]] = None, max_clusters: int = 5) -> None:
        self.print_node("root", self.root_node, 0, file, max_clusters)

    def print_node(self, token: str, node: Node, depth: int, file: Optional[IO[str]], max_clusters: int) -> None:
        out_str = '\t' * depth

        if depth == 0:
            out_str += f'<{token}>'
        elif depth == 1:
            if token.isdigit():
                out_str += f'<L={token}>'
            else:
                out_str += f'<{token}>'
        else:
            out_str += f'"{token}"'

        if len(node.cluster_ids) > 0:
            out_str += f" (cluster_count={len(node.cluster_ids)})"

        print(out_str, file=file)

        for token, child in node.key_to_child_node.items():
            self.print_node(token, child, depth + 1, file, max_clusters)

        for cid in node.cluster_ids[:max_clusters]:
            cluster = self.id_to_fused_templateCluster[cid]
            out_str = '\t' * (depth + 1) + str(cluster)
            #print(out_str, file=file)

    def get_content_as_tokens(self, content: str) -> Sequence[str]:
        content = content.strip()
        for delimiter in self.extra_delimiters:
            content = content.replace(delimiter, " ")
        content_tokens = content.split()
        return content_tokens

    def add_log_message(self, content: str) -> Tuple[LogCluster, str]:
        match_fused_cluster = None
        content_tokens = self.get_content_as_tokens(content)
        #print("input is:",content_tokens)
        if self.profiler:
            self.profiler.start_section("tree_search")
        match_tree_template_cluster = self.tree_search(self.root_node, content_tokens, self.sim_th, False)
        #print("tree_search return is:", match_tree_template_cluster)
        if self.profiler:
            self.profiler.end_section()

        # Match no existing log cluster
        if match_tree_template_cluster is None:
            #if self.profiler:
                #self.profiler.start_section("cluster_not_exist, build_templateCluster_with_input_log")
            match_fused_cluster = self.build_templateCluster_with_input_log(content_tokens)
            #if self.profiler:
                #self.profiler.end_section()
            update_type = "created/none"
        # Add the new log message to the existing cluster
        else:
            new_template_tokens = self.create_template(content_tokens, match_tree_template_cluster.get_tokens())
            #print(new_template_tokens)
            if tuple(new_template_tokens) == match_tree_template_cluster.get_tokens():
                update_type = "none"
                if self.profiler:
                    self.profiler.start_section("cluster_exist, get_fused_templateCluster")
                match_fused_cluster = self.get_fused_templateCluster(len(new_template_tokens), match_tree_template_cluster)
                if self.profiler:
                    self.profiler.end_section()
            else:
                match_tree_template_cluster.set_template(tuple(new_template_tokens))
                update_type = "changed"
                match_fused_cluster = self.update_fused_templateClusters(len(new_template_tokens), match_tree_template_cluster)
            match_tree_template_cluster.size += 1
            # Touch cluster to update its state in the cache.
            # noinspection PyStatementEffect
            self.id_to_tree_templateCluster[match_tree_template_cluster.cluster_id]
            #self.id_to_fused_templateCluster[match_fused_cluster.cluster_id]
            #if self.profiler:
                #self.profiler.end_section()

        #print("match_fused_cluster is:", match_fused_cluster.get_tokens())
        return match_fused_cluster, update_type

    def get_total_cluster_size(self) -> int:
        size = 0
        for c in self.id_to_fused_templateCluster.values():
            size += cast(LogCluster, c).size
        return size

    def get_clusters_ids_for_seq_len(self, seq_fir: Union[int, str]) -> Collection[int]:
        """
        seq_fir: int/str - the first token of the sequence
        Return all clusters with the specified count of tokens
        """

        def append_clusters_recursive(node: Node, id_list_to_fill: MutableSequence[int]) -> None:
            id_list_to_fill.extend(node.cluster_ids)
            for child_node in node.key_to_child_node.values():
                append_clusters_recursive(child_node, id_list_to_fill)

        cur_node = self.root_node.key_to_child_node.get(str(seq_fir))

        # no template with same token count
        if cur_node is None:
            return []

        target: MutableSequence[int] = []
        append_clusters_recursive(cur_node, target)
        return target

    @abstractmethod
    def tree_search(self,
                    root_node: Node,
                    tokens: Sequence[str],
                    sim_th: float,
                    include_params: bool) -> Optional[LogCluster]:
        ...

    @abstractmethod
    def add_seq_to_prefix_tree(self, root_node: Node, cluster: TemplateCluster) -> None:
        ...

    @abstractmethod
    def get_seq_distance(self, seq1: Sequence[str], seq2: Sequence[str], include_params: bool) -> Tuple[float, int]:
        ...

    @abstractmethod
    def create_template(self, seq1: Sequence[str], seq2: Sequence[str]) -> Sequence[str]:
        ...

    @abstractmethod
    def match(self, content: str, full_search_strategy: str = "never") -> Optional[LogCluster]:
        ...

    @abstractmethod
    def update_fused_templateClusters(self, in_log_cluster_id_pair: tuple,
                                             in_templateCluster: LogCluster) -> Optional[LogCluster]:
        ...

    @abstractmethod
    def get_fused_templateCluster_by_intersect_logCluster(self, in_log_cluster_id_pair: tuple) -> Optional[LogCluster]:
        ...

class Scope(ScopeBase):

    def get_cluster_ids_from_comparable_log_clusters(self, length: int) -> list:
        if length in self.comparable_logClusters:
            return [logCluster.cluster_id for logCluster in self.comparable_logClusters[length]]
        else:
            return []

    def add_log_into_comparable_logClusters(self, tokens: str) -> Optional[Cluster]:
        length = len(tokens)
        if length not in self.comparable_logClusters:
            self.comparable_logClusters[length] = []
        self.logCluster_counter += 1
        logCluster = LogCluster(tokens, self.logCluster_counter)
        self.comparable_logClusters[length].append(logCluster)
        self.id_to_logCluster[logCluster.cluster_id] = logCluster
        return logCluster

    def delete_cluster_from_comparable_log_clusters(self, length: int, cluster: Cluster) -> None:
          if length in self.comparable_logClusters:
              self.comparable_logClusters[length].remove(cluster)
              self.id_to_logCluster.pop(cluster.cluster_id, None)

    def get_fused_templateCluster_by_intersect_logCluster(self, length, log_cluster_id_pair: tuple) -> Optional[Cluster]:
        in_logClusterIdSet = set(log_cluster_id_pair)
        if length in self.logClstId_to_templateClstId:
            for id in in_logClusterIdSet:
                if id in self.logClstId_to_templateClstId[length]:
                    out_templateClst = self.id_to_fused_templateCluster[self.logClstId_to_templateClstId[length][id]]
                    out_templateClst.size += 1
                    return out_templateClst

    def get_fused_templateCluster(self, length, templateCluster: TemplateCluster) -> Optional[TemplateCluster]:
        return self.get_fused_templateCluster_by_intersect_logCluster(length, templateCluster.log_cluster_id_pair)

    def update_fused_templateClusters(self, length, in_templateCluster: TemplateCluster) -> Optional[TemplateCluster]:
        in_logClusterIdSet = set(in_templateCluster.log_cluster_id_pair)

        id_existed = False
        out_templateClstId = None
        if length in self.logClstId_to_templateClstId:
            for id in in_logClusterIdSet:
                if id in self.logClstId_to_templateClstId[length].keys():
                    id_existed = True
                    out_templateClstId = self.logClstId_to_templateClstId[length][id]
                    break
        if id_existed is True:
            out_templateClst = self.id_to_fused_templateCluster[out_templateClstId]
            out_templateClst.set_template(self.create_template(in_templateCluster.get_tokens(), out_templateClst.get_tokens()))
            out_templateClst.size += 1

            for id in in_logClusterIdSet:
                if id not in self.logClstId_to_templateClstId.keys():
                    self.logClstId_to_templateClstId[length][id] = out_templateClstId

            return out_templateClst

            """         if length not in self.logClusterIds_to_templateCluster:
                          self.logClusterIds_to_templateCluster[length] = []
                      for logClusterDict in self.logClusterIds_to_templateCluster[length]:
                          for out_logClusterIdSet, out_templateClst in logClusterDict.items():
                              #print("update_fused_templateClusters, existing:", out_templateClst.get_tokens())
                              if in_logClusterIdSet & set(out_logClusterIdSet):
                                  out_logClusterIdSet = set(out_logClusterIdSet) | in_logClusterIdSet
                                  self.profiler.start_section("build_templateCluster_with_input_log, 33333333333")
                                  out_templateClst.set_template(self.create_template(in_templateCluster.get_tokens(), out_templateClst.get_tokens()))
                                  out_templateClst.size += 1
                                  #print("update_fused_templateClusters, new template:", out_templateClst.get_tokens())
                                  self.profiler.end_section()
                                  return out_templateClst
                      #self.profiler.end_section() """
        else:
            self.profiler.start_section("build_templateCluster_with_input_log, 444444444444")
            new_templateCluster = self.create_fused_layer_template_cluster(in_templateCluster)
            new_templateCluster.size += 1
            #self.logClusterIds_to_templateCluster[length].append({tuple(in_logClusterIdSet): new_templateCluster})
            for logClusterId in in_logClusterIdSet:
                self.logClstId_to_templateClstId[length][logClusterId] = new_templateCluster.cluster_id
            self.profiler.end_section()
            return new_templateCluster

    def find_matched_cluster_from_comparable_logClusters(self, length, content_tokens: str) -> Optional[Cluster]:
        cluster_ids = self.get_cluster_ids_from_comparable_log_clusters(length)
        matched_log_cluster = self.fast_match(cluster_ids, content_tokens, self.sim_th, False, ClusterType.LOG)
        return matched_log_cluster

    def build_templateCluster_into_tree(self, a_logCluster: LogCluster, b_logCluster: LogCluster) -> Optional[Cluster]:
            templateCluster = self.create_tree_layer_template_cluster(a_logCluster, b_logCluster)
            self.add_seq_to_prefix_tree(self.root_node, templateCluster)
            return templateCluster

    def build_templateCluster_into_tree_with_own_logCluster(self, logCluster: LogCluster) -> Optional[Cluster]:
            return self.build_templateCluster_into_tree(logCluster, logCluster)

    def build_templateCluster_into_tree_with_matched_logCluster(self, \
                                                                matched_log_cluster: LogCluster, new_log_cluster: LogCluster) -> Optional[Cluster]:
            return self.build_templateCluster_into_tree(matched_log_cluster, new_log_cluster)

    def build_templateCluster_with_input_log(self, content_tokens: str) ->Optional[Cluster]:
        length = len(content_tokens)
        self.profiler.start_section("build_templateCluster_with_input_log, find_matched_cluster_from_comparable_logClusters")
        matched_log_cluster = self.find_matched_cluster_from_comparable_logClusters(length, content_tokens)
        self.profiler.end_section()
        # no matter template is build or not, add this log into comparable log clusters
        new_log_cluster = self.add_log_into_comparable_logClusters(content_tokens)
        if matched_log_cluster is not None:
            self.profiler.start_section("build_templateCluster_with_input_log, 11111111111")
            #1. create templateCluster based matched logCluster and input logCluster and add into tree
            templateCluster = self.build_templateCluster_into_tree_with_matched_logCluster(matched_log_cluster, new_log_cluster)
            #2. remove matched logCluster from comparable logClusters
            self.delete_cluster_from_comparable_log_clusters(length, matched_log_cluster)
            self.profiler.end_section()
        else:
            self.profiler.start_section("build_templateCluster_with_input_log, 2222222222222")
            templateCluster = self.build_templateCluster_into_tree_with_own_logCluster(new_log_cluster)
            self.profiler.end_section()

        #3. merge new template with legacy one based on its log cluster id pair
        fusedTemplateCluster = self.update_fused_templateClusters(length, templateCluster)
        return fusedTemplateCluster

    def tree_search(self,
                    root_node: Node,
                    tokens: Sequence[str],
                    sim_th: float,
                    include_params: bool) -> Optional[LogCluster]:

        # at first level, children are grouped by token (word) count
        token_count = len(tokens)
        cur_node = root_node.key_to_child_node.get(str(token_count))

        # no template with same token count yet
        if cur_node is None:
            return None

        # handle case of empty log string - return the single cluster in that group
        if token_count == 0:
            return self.id_to_tree_templateCluster.get(cur_node.cluster_ids[0])

        # find the leaf node for this log - a path of nodes matching the first N tokens (N=tree depth)
        for token in tokens:
            key_to_child_node = cur_node.key_to_child_node
            cur_node = key_to_child_node.get(token)
            if cur_node is None:  # no exact next token exist, try wildcard node
                cur_node = key_to_child_node.get(self.param_str)
            if cur_node is None:  # no wildcard node exist
                return None
            # token is matched:
            if cur_node.node_type == NodeType.LEAF:
                break
            #print("branch match:" f'{token} is matched')

        # get best match among all clusters with same prefix, or None if no match is above sim_th
        #print("branch is matched, go to fast_match")
        cluster = self.fast_match(cur_node.cluster_ids, tokens, sim_th, include_params)
        return cluster

    def add_seq_to_prefix_tree(self, root_node: Node, cluster: TemplateCluster) -> None:
        token_count = len(cluster.get_tokens())
        token_count_str = str(token_count)
        if token_count_str not in root_node.key_to_child_node:
            first_layer_node = Node(NodeType.INTERMEDIATE)
            root_node.key_to_child_node[token_count_str] = first_layer_node
        else:
            first_layer_node = root_node.key_to_child_node[token_count_str]

        cur_node = first_layer_node

        # handle case of empty log string
        if token_count == 0:
            cur_node.cluster_ids = [cluster.cluster_id]
            return

        current_depth = 1
        for token in cluster.get_tokens():
            #print("max_node_depth={}, cur_depth={}".format(self.max_node_depth, current_depth))
            # if at max depth or this is last token in template - add current log cluster to the leaf node
            if current_depth >= self.max_node_depth or current_depth >= token_count:
                # clean up stale clusters before adding a new one.
                new_cluster_ids = []
                for cluster_id in cur_node.cluster_ids:
                    if cluster_id in self.id_to_tree_templateCluster:
                        new_cluster_ids.append(cluster_id)
                new_cluster_ids.append(cluster.cluster_id)
                cur_node.cluster_ids = new_cluster_ids
                cur_node.node_type = NodeType.LEAF
                #print("cur_node.cluster_ids:", cur_node.cluster_ids)
                break

            # if token not matched in this layer of existing tree.
            if token not in cur_node.key_to_child_node:
                if self.parametrize_numeric_tokens and self.has_numbers(token):
                    if self.param_str not in cur_node.key_to_child_node:
                        new_node = Node(NodeType.INTERMEDIATE)
                        cur_node.key_to_child_node[self.param_str] = new_node
                        cur_node = new_node
                    else:
                        cur_node = cur_node.key_to_child_node[self.param_str]

                else:
                    if self.param_str in cur_node.key_to_child_node:
                        if len(cur_node.key_to_child_node) < self.max_children:
                            new_node = Node(NodeType.INTERMEDIATE)
                            cur_node.key_to_child_node[token] = new_node
                            cur_node = new_node
                        else:
                            cur_node = cur_node.key_to_child_node[self.param_str]
                    else:
                        if len(cur_node.key_to_child_node) + 1 < self.max_children:
                            new_node = Node(NodeType.INTERMEDIATE)
                            cur_node.key_to_child_node[token] = new_node
                            cur_node = new_node
                        elif len(cur_node.key_to_child_node) + 1 == self.max_children:
                            new_node = Node(NodeType.INTERMEDIATE)
                            cur_node.key_to_child_node[self.param_str] = new_node
                            cur_node = new_node
                        else:
                            cur_node = cur_node.key_to_child_node[self.param_str]

            # if the token is matched
            else:
                cur_node = cur_node.key_to_child_node[token]
            #print("add_seq_to_prefix_tree: add token is:", token)
            current_depth += 1

    # seq1 is a template, seq2 is the log to match
    def get_seq_distance(self, seq1: Sequence[str], seq2: Sequence[str], include_params: bool) -> Tuple[float, int]:
        assert len(seq1) == len(seq2)

        # sequences are empty - full match
        if len(seq1) == 0:
            return 1.0, 0

        sim_tokens = 0
        param_count = 0

        for token1, token2 in zip(seq1, seq2):
            if token1 == self.param_str:
                param_count += 1
                continue
            if token1 == token2:
                sim_tokens += 1

        if include_params:
            sim_tokens += param_count

        ret_val = float(sim_tokens) / len(seq1)

        return ret_val, param_count

    def create_template(self, seq1: Sequence[str], seq2: Sequence[str]) -> Sequence[str]:
        """
        Loop through two sequences and create a template sequence that
        replaces unmatched tokens with the parameter string.

        :param seq1: first sequence
        :param seq2: second sequence
        :return: template sequence with param_str in place of unmatched tokens
        """
        assert len(seq1) == len(seq2)
        return [token2 if token1 == token2 else self.param_str for token1, token2 in zip(seq1, seq2)]

    def create_tree_layer_template_cluster(self, old: LogCluster, new: LogCluster) -> Optional[TemplateCluster]:
        self.tree_templateClusters_cnt += 1
        template = self.create_template(old.get_tokens(), new.get_tokens())
        IdPair = (old.cluster_id, new.cluster_id)
        templateCluster = TemplateCluster(template, self.tree_templateClusters_cnt, IdPair)
        self.id_to_tree_templateCluster[templateCluster.cluster_id] = templateCluster
        #print("create_tree_layer_template_cluster, template is:", templateCluster.log_cluster_id_pair)
        return templateCluster

    def create_fused_layer_template_cluster(self, in_templateCluster: TemplateCluster) -> Optional[TemplateCluster]:
        self.fused_templateClusters_cnt += 1
        templateCluster = TemplateCluster(in_templateCluster.get_tokens(), self.fused_templateClusters_cnt)
        self.id_to_fused_templateCluster[templateCluster.cluster_id] = templateCluster
        return templateCluster

    def match(self, content: str, full_search_strategy: str = "never") -> Optional[LogCluster]:
        """
        Match log message against an already existing cluster.
        Match shall be perfect (sim_th=1.0).
        New cluster will not be created as a result of this call, nor any cluster modifications.

        :param content: log message to match
        :param full_search_strategy: when to perform full cluster search.
            (1) "never" is the fastest, will always perform a tree search [O(log(n)] but might produce
            false negatives (wrong mismatches) on some edge cases;
            (2) "fallback" will perform a linear search [O(n)] among all clusters with the same token count, but only in
            case tree search found no match.
            It should not have false negatives, however tree-search may find a non-optimal match with
            more wildcard parameters than necessary;
            (3) "always" is the slowest. It will select the best match among all known clusters, by always evaluating
            all clusters with the same token count, and selecting the cluster with perfect all token match and least
            count of wildcard matches.
        :return: Matched cluster or None if no match found.
        """

        assert full_search_strategy in ["always", "never", "fallback"]

        required_sim_th = 1.0
        content_tokens = self.get_content_as_tokens(content)

        # consider for future improvement:
        # It is possible to implement a recursive tree_search (first try exact token match and fallback to
        # wildcard match). This will be both accurate and more efficient than the linear full search
        # also fast match can be optimized when exact match is required by early
        # quitting on less than exact cluster matches.
        def full_search() -> Optional[LogCluster]:
            all_ids = self.get_clusters_ids_for_seq_len(len(content_tokens))
            cluster = self.fast_match(all_ids, content_tokens, required_sim_th, include_params=True)
            return cluster

        if full_search_strategy == "always":
            return full_search()

        match_cluster = self.tree_search(self.root_node, content_tokens, required_sim_th, include_params=True)
        if match_cluster is not None:
            return match_cluster

        if full_search_strategy == "never":
            return None

        return full_search()
