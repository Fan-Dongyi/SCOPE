# SPDX-License-Identifier: MIT
# This file implements the TOOLS algorithm for log parsing.

from abc import ABC, abstractmethod
from typing import cast, Collection, IO, Iterable, MutableMapping, MutableSequence, Optional, Sequence, Tuple, \
    TYPE_CHECKING, TypeVar, Union
from enum import Enum
from cachetools import LRUCache, Cache

from simple_profiler import Profiler, NullProfiler
from scope import ScopeBase
from collections import defaultdict


class Template():
    def __init__(self, templateTokens: Iterable[str] = "This is a default log template", templateId: int = 0) -> None:
      self.templateTokens = templateTokens
      self.templateId = templateId
      self.matchedLogSize = 1 # there is log matched this template when it's created, so 1 by default

    def getTemplateStr(self) -> str:
        return ' '.join(self.templateTokens)

    def getTemplate(self) -> str:
        return self.templateTokens

    def setTemplate(self, templateTokens: Iterable[str]) -> None:
        self.templateTokens = templateTokens

    def increaseMatchedLogSize(self) -> None:
        self.matchedLogSize += 1

    def getMatchedLogSize(self) -> int:
        return self.matchedLogSize


_T = TypeVar("_T")
if TYPE_CHECKING:
    class _LRUCache(LRUCache[int, Optional[Template]]):
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

    def get(self, key: int, _: Union[Optional[Template], _T] = None) -> Optional[Template]:
        """
        Returns the value of the item with the specified key without updating
        the cache eviction algorithm.
        """
        return Cache.__getitem__(self, key)

class NodeType(Enum):
    ROOT = 1
    DIRECTION = 2
    INTERMEDIATE = 3
    LEAF = 4

class SequenceType(Enum):
    FORWARD = 1
    REVERSE = 2

class Node():
    __slots__ = ["nodeType", "keyToChildNode", "templateIds"]

    def __init__(self, nodeType: NodeType) -> None:
        self.nodeType: NodeType = nodeType
        self.keyToChildNode: MutableMapping[str, Node] = {}
        self.templateIds: Sequence[int] = set()


class Tools(ScopeBase):
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

        self.idToTemplateCluster: MutableMapping[int, Template] = {}
        self.lengthToTemplateIds = defaultdict(list)
        self.templateId = 0

    def getNewTemplateId(self) -> int:
        self.templateId += 1
        return self.templateId

    @property
    def clusters(self) -> Collection[Template]:
        return cast(Collection[Template], self.idToTemplateCluster.values())

    @staticmethod
    def has_numbers(s: Iterable[str]) -> bool:
        return any(char.isdigit() for char in s)

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

        if len(node.templateIds) > 0:
            out_str += f" (cluster_count={len(node.templateIds)})"

        #print(out_str, file=file)

        for token, child in node.keyToChildNode.items():
            self.print_node(token, child, depth + 1, file, max_clusters)

        for cid in node.templateIds[:max_clusters]:
            cluster = self.idToTemplateCluster[cid]
            out_str = '\t' * (depth + 1) + str(cluster)
            #print(out_str, file=file)

    def get_content_as_tokens(self, content: str) -> Sequence[str]:
        content = content.strip()
        for delimiter in self.extra_delimiters:
            content = content.replace(delimiter, " ")
        content_tokens = content.split()
        return content_tokens



    def get_total_cluster_size(self) -> int:
        size = 0
        for c in self.idToTemplateCluster.values():
            size += cast(Te, c).size
        return size

    def get_clusters_ids_for_seq_len(self, seq_fir: Union[int, str]) -> Collection[int]:
        """
        seq_fir: int/str - the first token of the sequence
        Return all clusters with the specified count of tokens
        """

        def append_clusters_recursive(node: Node, id_list_to_fill: MutableSequence[int]) -> None:
            id_list_to_fill.extend(node.templateIds)
            for child_node in node.keyToChildNode.values():
                append_clusters_recursive(child_node, id_list_to_fill)

        cur_node = self.root_node.keyToChildNode.get(str(seq_fir))

        # no template with same token count
        if cur_node is None:
            return []

        target: MutableSequence[int] = []
        append_clusters_recursive(cur_node, target)
        return target

    def add_seq_to_prefix_tree(self, root_node: Node, cluster: Template, seqType: SequenceType) -> None:
        tokens = cluster.getTemplate()
        token_count = len(tokens)
        #print("existing", tokens)
        token_count_str = str(token_count)
        token_seqType_str = str(int(seqType.value))
        if token_count_str not in root_node.keyToChildNode:
            first_layer_node = Node(NodeType.DIRECTION)
            root_node.keyToChildNode[token_count_str] = first_layer_node
            #print("create LENGTH node:", token_count_str)
        else:
            first_layer_node = root_node.keyToChildNode[token_count_str]
            #print("existing LENGTH node:", token_count_str)

        if  token_seqType_str not in first_layer_node.keyToChildNode:
            sec_layer_node = Node(NodeType.INTERMEDIATE)
            first_layer_node.keyToChildNode[token_seqType_str] = sec_layer_node
            #print("create DIR node:", token_seqType_str)
        else:
            sec_layer_node = first_layer_node.keyToChildNode[token_seqType_str]
            #print("existing DIR node:", token_seqType_str)

        cur_node = sec_layer_node

        # handle case of empty log string
        if token_count == 0:
            cur_node.templateIds = [cluster.templateId]
            return

        tokens = tokens[::1] if seqType == SequenceType.FORWARD else tokens[::-1]

        #tokens = tokens.split()
        #print("tokens:", tokens)

        def get_half_tokens(tokens):
            if token_count % 2 == 0:
                return tokens[:token_count // 2+1]
            else:
                return tokens[:(token_count + 1) // 2]
        tokens = get_half_tokens(tokens)
        current_depth = 0
        for token in tokens:
            #print("max_node_depth={}, cur_depth={}".format(self.max_node_depth, current_depth))
            # if at max depth or this is last token in template - add current log cluster to the leaf node
            if current_depth >= self.max_node_depth or current_depth >= len(tokens):
                # clean up stale clusters before adding a new one.
                new_cluster_ids = set()
                for cluster_id in cur_node.templateIds:
                    if cluster_id in self.idToTemplateCluster:
                        new_cluster_ids.add(cluster_id)
                new_cluster_ids.add(cluster.templateId)
                cur_node.templateIds = new_cluster_ids
                cur_node.nodeType = NodeType.LEAF
                #print("cur_node.templateIds:", cur_node.templateIds)
                break

            # if token not matched in this layer of existing tree.
            if token not in cur_node.keyToChildNode:
                if self.parametrize_numeric_tokens and self.has_numbers(token):
                    if self.param_str not in cur_node.keyToChildNode:
                        new_node = Node(NodeType.INTERMEDIATE)
                        cur_node.keyToChildNode[self.param_str] = new_node
                        cur_node = new_node
                    else:
                        cur_node = cur_node.keyToChildNode[self.param_str]

                else:
                    if self.param_str in cur_node.keyToChildNode:
                        if len(cur_node.keyToChildNode) < self.max_children:
                            new_node = Node(NodeType.INTERMEDIATE)
                            cur_node.keyToChildNode[token] = new_node
                            cur_node = new_node
                        else:
                            cur_node = cur_node.keyToChildNode[self.param_str]
                    else:
                        if len(cur_node.keyToChildNode) + 1 < self.max_children:
                            new_node = Node(NodeType.INTERMEDIATE)
                            cur_node.keyToChildNode[token] = new_node
                            cur_node = new_node
                        elif len(cur_node.keyToChildNode) + 1 == self.max_children:
                            new_node = Node(NodeType.INTERMEDIATE)
                            cur_node.keyToChildNode[self.param_str] = new_node
                            cur_node = new_node
                        else:
                            cur_node = cur_node.keyToChildNode[self.param_str]

            # if the token is matched
            else:
                cur_node = cur_node.keyToChildNode[token]
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

    def match(self, content: str, full_search_strategy: str = "never") -> Optional[Template]:
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
        def full_search() -> Optional[Template]:
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

    def fast_match(self,
                   templateIds: Collection[int],
                   tokens: Sequence[str],
                   sim_th: float,
                   include_params: bool) -> Optional[Template]:
        """
        Find the best match for a log message (represented as tokens) versus a list of clusters
        :param templateIds: List of clusters to match against (represented by their IDs)
        :param tokens: the log message, separated to tokens.
        :param sim_th: minimum required similarity threshold (None will be returned in no clusters reached it)
        :param include_params: consider tokens matched to wildcard parameters in similarity threshold.
        :return: Best match cluster or None
        """
        match_cluster = None

        max_sim: Union[int, float] = -1
        max_param_count = -1
        max_cluster = None

        #print("templateIds:", templateIds)

        for id in templateIds:
            # Try to retrieve cluster from cache with bypassing eviction
            # algorithm as we are only testing candidates for a match.
            cluster = self.idToTemplateCluster.get(id)
            if cluster is None:
                continue
            cur_sim, param_count = self.get_seq_distance(cluster.getTemplate(), tokens, include_params)
            if cur_sim > max_sim or (cur_sim == max_sim and param_count > max_param_count):
                max_sim = cur_sim
                max_param_count = param_count
                max_cluster = cluster

        if max_sim >= sim_th:
            match_cluster = max_cluster
            # print("max_sim:", max_sim, "sim_th:", sim_th)
            # print("fast_match: tree template:", match_cluster.getTemplate())
            # print("fast_match: input tokens:", tokens)
        return match_cluster

    def findMatchedTemplateFromPool(self, length, tokens: Sequence[str]) -> Optional[Template]:
        if length not in self.lengthToTemplateIds:
            return None
        templateIds = self.lengthToTemplateIds.get(length)
        #print("length:", length, "size is:", len(templateIds))
        matchedTemplate = self.fast_match(templateIds, tokens, self.sim_th, False)
        return matchedTemplate

    def updateTemplateOfPool(self, template: Template, newTemplateStr: Sequence[str]) -> None:
        template.setTemplate(newTemplateStr)
        template.increaseMatchedLogSize()
        # lst = self.lengthToTemplateIds[len(newTemplateStr)]
        # if template.templateId in lst:
        #     lst.append(lst.pop(lst.index(template.templateId)))

    def findMatchedTemplateFromTree(self,
                    root_node: Node,
                    tokens: Sequence[str],
                    sim_th: float,
                    include_params: bool) -> tuple[Template, Template]:
        fw = self.tree_search(root_node, tokens, SequenceType.FORWARD, sim_th, include_params)
        rv =  self.tree_search(root_node, tokens, SequenceType.REVERSE, sim_th, include_params)
        return fw, rv

    def tree_search(self,
                    root_node: Node,
                    tokens: Sequence[str],
                    seq_type: SequenceType,
                    sim_th: float,
                    include_params: bool) -> Optional[Template]:

        # at first level, children are grouped by token (word) count
        token_count = len(tokens)
        #print("token_count:", token_count)
        cur_node = root_node.keyToChildNode.get(str(token_count))
        #print("length node:", cur_node)
        # no template with same token count yet
        if cur_node is None:
            return None

        cur_node = cur_node.keyToChildNode.get(str(int(seq_type.value)))
        #print("direction node:", cur_node)
        # no template with matched dirction sequence yet
        if cur_node is None:
            return None
        #print("cur_node:", cur_node)
        # handle case of empty log string - return the single cluster in that group
        if token_count == 0:
            return self.idToTemplateCluster.get(cur_node.templateIds[0])

        # find the leaf node for this log - a path of nodes matching the first N tokens (N=tree depth)
        tokensToMatch = tokens[::1] if seq_type == SequenceType.FORWARD else tokens[::-1]
        #print("tokens:", tokens)

        for token in tokensToMatch:
            keyToChildNode = cur_node.keyToChildNode
            cur_node = keyToChildNode.get(token)
            if cur_node is None:  # no exact next token exist, try wildcard node
                cur_node = keyToChildNode.get(self.param_str)
            if cur_node is None:  # no wildcard node exist
                return None
            # token is matched:
            if cur_node.nodeType == NodeType.LEAF:
                break
            #print("branch match:" f'{token} is matched')

        # get best match among all clusters with same prefix, or None if no match is above sim_th
        #print("branch is matched, go to fast_match")
        cluster = self.fast_match(cur_node.templateIds, tokens, sim_th, include_params)
        return cluster

    def buildTemplateWithInputLog(self, length, tokens: Sequence[str]) -> Optional[Template]:
        template = Template(tokens, self.getNewTemplateId())
        self.idToTemplateCluster[template.templateId] = template
        #self.lengthToTemplateIds[length].append(template.templateId)
        self.lengthToTemplateIds[length].insert(0, template.templateId)
        #print("length:", length, "str is:", tokens)
        return template

    def addTemplateSeqToPrefixTree(self, root_node: Node, template: Template) -> None:
        self.add_seq_to_prefix_tree(root_node, template, SequenceType.FORWARD)
        self.add_seq_to_prefix_tree(root_node, template, SequenceType.REVERSE)

    def add_log_message(self, content: str) -> Tuple[Template, str]:
        content_tokens = self.get_content_as_tokens(content)
        length = len(content_tokens)
        #print("input is:",content_tokens)
        if self.profiler:
            self.profiler.start_section("findMatchedTemplateFromTree")
        (fwSeqMatchedTemplate, RvSeqMatchedTemplate) = self.findMatchedTemplateFromTree(self.root_node, content_tokens, self.sim_th, False)
        #print("tree_search return is:(", fwSeqMatchedTemplate, RvSeqMatchedTemplate, ")")
        if self.profiler:
            self.profiler.end_section()

        # Match no existing template
        if fwSeqMatchedTemplate is None and RvSeqMatchedTemplate is None: # both forward and reverse sequence don't have matched template
            if self.profiler:
                self.profiler.start_section("Match no existing template, findMatchedTemplateFromPool")

            matchedPoolTemplate = self.findMatchedTemplateFromPool(length, content_tokens)
            if self.profiler:
                self.profiler.end_section()
            if matchedPoolTemplate is None: # it's a new message which don't have template in pool yet
                matchedPoolTemplate = self.buildTemplateWithInputLog(length, content_tokens)
                update_type = "created"
            else: # similar template is found in pool but not in tree, need update template and add to tree
                newTemplateStr = self.create_template(content_tokens, matchedPoolTemplate.getTemplate())
                self.updateTemplateOfPool(matchedPoolTemplate, newTemplateStr)
                update_type = "updated"
            self.addTemplateSeqToPrefixTree(self.root_node, matchedPoolTemplate)


        else: # Match existing template at least one direction of tree
            if self.profiler:
                self.profiler.start_section("Match any existing template")

            if fwSeqMatchedTemplate is not None and RvSeqMatchedTemplate is not None:
                if(fwSeqMatchedTemplate.templateId != RvSeqMatchedTemplate.templateId):
                    print("input is:",content_tokens)
                    print("fwSeqMatchedTemplate.templateId:", fwSeqMatchedTemplate.templateId)
                    print("fw template:", fwSeqMatchedTemplate.getTemplateStr())
                    print("RvSeqMatchedTemplate.templateId:", RvSeqMatchedTemplate.templateId)
                    print("Rv template:", RvSeqMatchedTemplate.getTemplateStr())
                assert(fwSeqMatchedTemplate.templateId == RvSeqMatchedTemplate.templateId)
                matchedPoolTemplate = fwSeqMatchedTemplate
            elif fwSeqMatchedTemplate is not None:
                matchedPoolTemplate = fwSeqMatchedTemplate
            else:
                matchedPoolTemplate = RvSeqMatchedTemplate

            newTemplateStr = self.create_template(content_tokens, matchedPoolTemplate.getTemplate())
            if newTemplateStr != matchedPoolTemplate.getTemplate():
                update_type = "updated"
                self.updateTemplateOfPool(matchedPoolTemplate, newTemplateStr)
                self.addTemplateSeqToPrefixTree(self.root_node, matchedPoolTemplate)
            else:
                update_type = "none"
                matchedPoolTemplate.increaseMatchedLogSize()
            if self.profiler:
                self.profiler.end_section()

        return matchedPoolTemplate, update_type

    def get_fused_templateCluster_by_intersect_logCluster(self, length, log_cluster_id_pair: tuple,
                                                              in_templateCluster: Template) -> Optional[Template]:

        return in_templateCluster


    def update_fused_templateClusters(self, in_log_cluster_id_pair: Tuple, in_templateCluster: Template) -> Template | None:
        return super().update_fused_templateClusters(in_log_cluster_id_pair, in_templateCluster)

