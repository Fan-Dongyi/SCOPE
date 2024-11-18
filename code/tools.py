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
import logging
import logging.config
import re
import nltk
import en_core_web_md
import math

nlp = en_core_web_md.load()
path = nltk.data.find('taggers/averaged_perceptron_tagger')
if path is None:
    exit(1)
else:
    print(path)
nltk.data.path.append(path)
#nltk.download('averaged_perceptron_tagger', download_dir=path)

#logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logging.basicConfig(filename="./tools.log", filemode="w", level=logging.INFO, format="%(levelname)s - %(message)s")
#logging.basicConfig(filename="./tools.log", filemode="w", level=logging.DEBUG, format="%(levelname)s - %(message)s")
logger = logging.getLogger() # default 'root' name is used

# Example log messages
#logger.debug("This is a DEBUG message")
#logger.info("This is an INFO message")
#logger.warning("This is a WARNING message")
#logger.error("This is an ERROR message")

class Template():
    def __init__(self, templateTokens: Iterable[str] = "This is a default log template", templateId: int = 0, isPosSupported: bool = False) -> None:
      self.templateId = templateId
      self.matchedLogSize = 1 # there is log matched this template when it's created, so 1 by default
      self.preemptedTokenSet = defaultdict(set) # {0:set(preempted token, <*>), 1:set(), 2:set(), ...} it stores the preempted token which would be assigned to <*> node to preempt existing node token in tree
      self.isPosSupported = isPosSupported
      self.setTemplate(templateTokens)
      self.setTokenPosTag(templateTokens)

    def getTemplateStr(self) -> str:
        return ' '.join(self.templateTokens)

    def getTemplateTokens(self) -> str:
        return self.templateTokens

    def getPreemptedTokenSet(self) -> MutableMapping[int, set]:
        return self.preemptedTokenSet

    def setTemplate(self, templateTokens: Iterable[str]) -> None:
        self.templateTokens = templateTokens
        for index, token in enumerate(templateTokens): # start from 0
            self.preemptedTokenSet[index].add(token)
            logger.debug("preemptedTokenSet[{}]:{}".format(index, self.preemptedTokenSet[index]))

    def increaseMatchedLogSize(self) -> None:
        self.matchedLogSize += 1

    def getMatchedLogSize(self) -> int:
        return self.matchedLogSize

    def getTokenPosTag(self) -> MutableMapping[int, str]:
        return self.tokenPosTag

    def setTokenPosTag(self, templateTokens: Iterable[str]) -> None:
        self.tokenPosTag = self.tokensPosTagger(templateTokens)

    def tokensPosTagger(self, tokens: Iterable[str]) -> Iterable[str]:
        if self.isPosSupported:
            lowercaseTokens = [element.lower() for element in tokens]
            posTagger = nltk.pos_tag(lowercaseTokens)
            tokenPosTag = [pos for token, pos in posTagger]
        else:
            tokenPosTag = ["unknown"] * len(tokens)
        logger.debug("template token: %s", tokens)
        logger.debug("tokenPosTag: %s", tokenPosTag)
        return tokenPosTag

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
    __slots__ = ["nodeType", "keyToChildNode", "templateIds", "tokensInWildcard"]

    def __init__(self, nodeType: NodeType) -> None:
        self.nodeType: NodeType = nodeType
        self.keyToChildNode: MutableMapping[str, Node] = {}
        self.templateIds: Sequence[int] = set()
        self.tokensInWildcard = set()


class Tools(ScopeBase):
    def __init__(self,
                 depth: int = 4,
                 sim_th: float = 0.4,
                 max_children: int = 100,
                 max_clusters: Optional[int] = None,
                 extra_delimiters: Sequence[str] = (),
                 profiler: Profiler = NullProfiler(),
                 param_str: str = "<*>",
                 parametrize_numeric_tokens: bool = True,
                 bi_tree_support: bool = False,
                 pos_support: bool = False) -> None:
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
        self.bi_tree_support = bi_tree_support
        self.pos_support = pos_support

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

    def split_string(self, s):
        # 定义要匹配的标点
        #punctuation_pattern = r'^[,.\!?[\];:]+|[,.\!?[\];:]+$'
        punctuation_pattern = r'^[,.\!?;:]+|[,.\!?;:]+$'

        # 使用正则表达式找到前后的标点符号
        match = re.match(punctuation_pattern, s)
        if match:
            # 如果匹配到了前面的标点，获取前面的标点符号
            front_punctuation = match.group(0)
        else:
            front_punctuation = ''

        # 去除前面的标点符号
        stripped_s = re.sub(r'^[,.\!?;:]+', '', s)

        # 使用正则表达式找到后面的标点符号
        match = re.search(punctuation_pattern, stripped_s)
        if match:
            # 如果匹配到了后面的标点，获取后面的标点符号
            back_punctuation = match.group(0)
            # 去除后面的标点符号
            stripped_s = re.sub(r'[,.\!?;:]+$', '', stripped_s)
        else:
            back_punctuation = ''

        # 初始化结果列表
        result = []

        # Case where both ':' and '=' are in the string
        if '=' in stripped_s and ':' in stripped_s:
            # Find the first occurrence and split accordingly
            if stripped_s.index('=') < stripped_s.index(':'):
                left, sep1, remainder = re.split(r"(=)", stripped_s, 1)
                mid, sep2, right = re.split(r"(:)", remainder, 1)
            else:
                left, sep1, remainder = re.split(r"(:)", stripped_s, 1)
                mid, sep2, right = re.split(r"(=)", remainder, 1)
            result.extend([left, sep1, mid, sep2, right])

        # Case where only '=' is present
        elif stripped_s.count('=') == 1:
            left, sep, right = re.split(r"(=)", stripped_s, 1)
            result.extend([left, sep, right])

        # Case where only ':' is present
        elif stripped_s.count(':') == 1:
            left, sep, right = re.split(r"(:)", stripped_s, 1)
            result.extend([left, sep, right])

        # Case where neither ':' nor '=' is present
        else:
            result.append(stripped_s)

        # 将前后的标点符号作为单独的 token 添加到结果中
        if front_punctuation:
            result.insert(0, front_punctuation)  # 添加开头的标点
        if back_punctuation:
            result.append(back_punctuation)  # 添加结尾的标点

        return result

    def param_str_split_string(self, s):
        pattern = r"(.*?)(<[^:]+>)(.*)"
        match = re.match(pattern, s)
        if 0: #match:
            parts = list(match.groups())
            parts = [p for p in parts if p]
            return parts
        else:
            return [s]

    def get_content_as_tokens(self, content: str) -> Sequence[str]:
        content = content.strip()
        #print("content: ", content)
        for delimiter in self.extra_delimiters:
            #print("delimeiter is: ", delimiter )
            content = content.replace(delimiter, " ")
        content_tokens = content.split()
        #print("content tokens: ", content_tokens)
        if (self.pos_support):
            new_tokens = []
            for token in content_tokens:
                parStrSplitTokens = self.param_str_split_string(token)
                new_tokens.extend(parStrSplitTokens)
            split_tokens = []
            for token in new_tokens:
                #print("token is====================== ", token)
                if (re.match(r"\d{1,3}(\.\d{1,3}){3}(:\d{1,5})?$", token) or
                    re.match(r"\d{1,2}:\d{2}(:\d{2})?$", token) or
                    re.match(r"^<[^:]+>$", token)):
                    split_tokens.append(token)
                else:
                    #print("need split token is====================== ", token)
                    # 将前后标点（如 , . ! ?）与中间内容分离；始终分离 = 和 :
                    punc_split_tokens = self.split_string(token)
                    # 去除空匹配并添加到新列表
                    #print("split token is============= ", split_tokens)
                    split_tokens.extend([t for t in punc_split_tokens if t])
            content_tokens = split_tokens
                # 匹配 IP 地址或时间格式不分隔的 token
        logger.debug("content tokens reg: %s", content_tokens)


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
        tokens = cluster.getTemplateTokens()
        preemptedTokenSet = cluster.getPreemptedTokenSet()
        token_count = len(tokens)
        logger.debug("existing %s", tokens)
        token_count_str = str(token_count)
        token_seqType_str = str(int(seqType.value))
        if token_count_str not in root_node.keyToChildNode:
            first_layer_node = Node(NodeType.DIRECTION)
            root_node.keyToChildNode[token_count_str] = first_layer_node
            logger.debug("create LENGTH node: %s", token_count_str)
        else:
            first_layer_node = root_node.keyToChildNode[token_count_str]
            logger.debug("existing LENGTH node: %s", token_count_str)

        if  token_seqType_str not in first_layer_node.keyToChildNode:
            sec_layer_node = Node(NodeType.INTERMEDIATE)
            first_layer_node.keyToChildNode[token_seqType_str] = sec_layer_node
            logger.debug("create DIR node: %s", token_seqType_str)
        else:
            sec_layer_node = first_layer_node.keyToChildNode[token_seqType_str]
            logger.debug("existing DIR node: %s", token_seqType_str)

        cur_node = sec_layer_node

        # handle case of empty log string
        if token_count == 0:
            cur_node.templateIds = [cluster.templateId]
            return

        tokens = tokens[::1] if seqType == SequenceType.FORWARD else tokens[::-1]

        #tokens = tokens.split()
        logger.debug("tokens: %s", tokens)

        def get_half_tokens(tokens):
            if token_count % 2 == 0:
                return tokens[:token_count // 2 + 1]
            else:
                return tokens[:(token_count + 1) // 2]
        tokens = get_half_tokens(tokens)

        current_depth = 1
        for index, token in enumerate(tokens):
            logger.debug("max_node_depth={}, cur_depth={}".format(self.max_node_depth, current_depth))
            index = index if seqType == SequenceType.FORWARD else token_count - index - 1
            if token == self.param_str:
                if self.param_str not in cur_node.keyToChildNode: # * node always can be added as room is reserved
                    new_node = Node(NodeType.INTERMEDIATE)
                    new_node.tokensInWildcard = preemptedTokenSet[index] # <*> node is new created, set node.tokensInWildcard = preempted tokens of template
                    logger.debug(f"preemptedTokenSet[{index}]: {preemptedTokenSet[index]}")
                    logger.debug("no <*>, update tokensInWildcard: %s", new_node.tokensInWildcard)
                    cur_node.keyToChildNode[self.param_str] = new_node
                    cur_node = new_node
                else:
                    new_node = cur_node.keyToChildNode[self.param_str]
                    new_node.tokensInWildcard = new_node.tokensInWildcard | preemptedTokenSet[index] # <*> node exists, add preempted tokens of template into node.tokensInWildcard
                    cur_node = new_node
                    logger.debug(" <*> exists, update tokensInWildcard: %s", new_node.tokensInWildcard)
            else:
                if token not in cur_node.keyToChildNode:
                    if self.parametrize_numeric_tokens and self.has_numbers(token): # it's a parameter token
                        if self.param_str not in cur_node.keyToChildNode:
                            new_node = Node(NodeType.INTERMEDIATE)
                            #new_node.tokensInWildcard.append(token) # <*> node is used to preempt existing token in tree
                            cur_node.keyToChildNode[self.param_str] = new_node
                            cur_node = new_node
                        else:
                            cur_node = cur_node.keyToChildNode[self.param_str] # <*> is selected if token is not in tree
                    else: # it's a normal token
                        if self.param_str in cur_node.keyToChildNode: # <*> node exists in tree, add token to it if room is available
                            if len(cur_node.keyToChildNode) < self.max_children:
                                new_node = Node(NodeType.INTERMEDIATE)
                                cur_node.keyToChildNode[token] = new_node
                                cur_node = new_node
                            else:
                                cur_node = cur_node.keyToChildNode[self.param_str] # if the number of children is full, use * node directly
                        else: # <*> node not exists in tree
                            if len(cur_node.keyToChildNode) + 1 < self.max_children: # there is room for new token if at least one room is reserved for <*>
                                new_node = Node(NodeType.INTERMEDIATE)
                                cur_node.keyToChildNode[token] = new_node
                                cur_node = new_node
                            else: # only 1 room left, only can add <*> node
                                new_node = Node(NodeType.INTERMEDIATE)
                                #new_node.tokensInWildcard.append(token) #<*> node is used to preempt existing token in tree
                                cur_node.keyToChildNode[self.param_str] = new_node
                                cur_node = new_node
                else: # if the token is matched
                    cur_node = cur_node.keyToChildNode[token]
            logger.debug("add_seq_to_prefix_tree: add token is: %s", token)
            current_depth += 1

            if current_depth >= self.max_node_depth or current_depth > len(tokens):
                # clean up stale clusters before adding a new one.
                logger.debug("end of token add")
                new_cluster_ids = set()
                for cluster_id in cur_node.templateIds:
                    if cluster_id in self.idToTemplateCluster:
                        new_cluster_ids.add(cluster_id)
                new_cluster_ids.add(cluster.templateId)
                cur_node.templateIds = new_cluster_ids
                cur_node.nodeType = NodeType.LEAF
                logger.debug("cur_node.templateIds: %s", cur_node.templateIds)
                break

    # seq1 is a template, seq2 is the log to match
    def get_seq_distance(self, seq1: Sequence[str], seq2: Sequence[str], \
                         preemptedTokens, tokenPosTag, include_params: bool) -> Tuple[float, int]:
        assert len(seq1) == len(seq2)

        # sequences are empty - full match
        if len(seq1) == 0:
            return 1.0, 0

        sim_tokens = 0
        param_count = 0

        nltk_static_pos_tags = [
            "CC",    # Coordinating conjunction
            #"CD",    # Cardinal number
            "DT",    # Determiner
            "EX",    # Existential there
            "FW",    # Foreign word
            "IN",    # Preposition or subordinating conjunction
            #"JJ",    # Adjective
            #"JJR",   # Adjective, comparative
            #"JJS",   # Adjective, superlative
            "LS",    # List item marker
            "MD",    # Modal
            #"NN",    # Noun, singular or mass
            #"NNS",   # Noun, plural
            #"NNP",   # Proper noun, singular
            #"NNPS",  # Proper noun, plural
            "PDT",   # Predeterminer
            "POS",   # Possessive ending
            "PRP",   # Personal pronoun
            "PRP$",  # Possessive pronoun
            "RB",    # Adverb
            "RBR",   # Adverb, comparative
            "RBS",   # Adverb, superlative
            "RP",    # Particle
            "SYM",   # Symbol
            "TO",    # to
            "UH",    # Interjection
            #"VB",    # Verb, base form
            #"VBD",   # Verb, past tense
            #"VBG",   # Verb, gerund or present participle
            #"VBN",   # Verb, past participle
            #"VBP",   # Verb, non-3rd person singular present
            #"VBZ",   # Verb, 3rd person singular present
            #"WDT",   # Wh-determiner
            #"WP",    # Wh-pronoun
            #"WP$",   # Possessive wh-pronoun
            #"WRB"    # Wh-adverb
        ]
        def is_determinated_static_part(posTag: str) -> bool:
            if any(i in posTag for i in nltk_static_pos_tags):
                return True
            else:
                return False
        logger.debug("Template to match: %s", seq1) # seq1 is the template

        def isAllAlphaCapital(token: str) -> bool:
            return all(char.isupper() for char in token if char.isalpha())

        def is_snake_or_camel(identifier):
            snake_case_pattern = r'^[a-z]+(_[a-z]+)+$'
            camel_case_pattern = r'^[a-zA-Z]+([A-Z][a-z]+)+$'
            return bool(re.match(snake_case_pattern, identifier) or re.match(camel_case_pattern, identifier))

        for index, (token1, token2) in enumerate(zip(seq1, seq2)):
            logger.debug("index: %s, template token: %s, input token: %s", index, token1, token2)
            if token1 == token2:
                sim_tokens += 1
            elif re.match(r"<[^:]+>", token1) or re.match(r"<[^:]+>", token2):
                if token2 in preemptedTokens[index]:
                    param_count += 1
            elif ((not self.has_numbers(token1) and not self.has_numbers(token2)) or (isAllAlphaCapital(token1) and isAllAlphaCapital(token2))) \
                        and not re.search(r"<[^:]+>", token1) and not re.search(r"<[^:]+>", token2):
                if len(seq1) == 1:
                    logger.debug(f"token: {token1}, {token2} is the only token, to be static part")
                    return 0.0, 0

                if 0: #self.pos_support:
                    lowercaseTokens = [element.lower() for element in seq2]
                    posTagger = nltk.pos_tag(lowercaseTokens)
                    destTokenPosTag = [pos for token, pos in posTagger]
                    logger.debug("destTokenPosTag: %s", destTokenPosTag)

                    if is_determinated_static_part(tokenPosTag[index]) or is_determinated_static_part(destTokenPosTag[index]):
                      logger.debug(f"static part is matched: {token1}: POS type: {tokenPosTag[index]} or {token2}: POS type: {destTokenPosTag[index]}")
                      return 0.0, 0
                    if tokenPosTag[index][0:1] != destTokenPosTag[index][0:1]:
                      logger.debug(f"token: {token1}, {token2} has different POS type: {tokenPosTag[index]} and {destTokenPosTag[index]}")
                      return 0.0, 0
                    elif("VB" in tokenPosTag[index] or "JJ" in tokenPosTag[index] or "NN" in tokenPosTag[index] or \
                          "VB" in destTokenPosTag[index] or "JJ" in destTokenPosTag[index] or "NN" in destTokenPosTag[index]):
                        if (index == 0 and (not tokenPosTag[index+1].startswith("VB") and not destTokenPosTag[index+1].startswith("VB")) \
                            and (not any(char in seq1[index] for char in ['_', '/']) and not any(char in seq2[index] for char in ['_', '/']))):
                            logger.debug(f"token: {token1}, {token2} is the first token and not followed by VB, not special string, to be static part")#1790
                            return 0.0, 0
                        if index>0 and (seq1[index-1] != ":" and seq1[index-1] != "=" and tokenPosTag[index-1] != "IN" and tokenPosTag[index-1] != "TO") \
                            and (seq2[index-1] != ":" and seq2[index-1] != "=" and destTokenPosTag[index-1] != "IN" and destTokenPosTag[index-1] != "TO") \
                            and ((not any(char in seq1[index] for char in ['_', '/']) and not any(char in seq2[index] for char in ['_', '/']))\
                            and (not self.has_numbers(token1) or not self.has_numbers(token2))) \
                            and not (seq1[index-1] == seq2[index-1] and (tokenPosTag[index-1].startswith("JJ") or destTokenPosTag[index-1].startswith("JJ"))):
                            logger.debug(f"token: {token1}, {token2} don't stay after IN or TO type, and not special string and not both having NUM, to be static part")#3215
                            return 0.0, 0
                        if index>0 and "VB" in tokenPosTag[index] and (tokenPosTag[index-1] == "IN" or tokenPosTag[index-1] == "TO"):
                            logger.debug(f"token: {token1}, {token2} is a verb after IN or TO, to be static part")#17 'to spawn'
                            return 0.0, 0
                        if index>0 and "NN" in destTokenPosTag[index] and "VB" in destTokenPosTag[index-1]:
                            logger.debug(f"token: {token1}, {token2} is a noun after verb, to be static part")# removing xxx
                            return 0.0, 0

                if is_snake_or_camel(token2) and is_snake_or_camel(token1):
                    logger.debug(f"token: {token1} and {token2} is snake or camel case, to be static part")
                    return 0.0, 0
                if index>0 and index<len(seq1)-1 and (seq2[index+1] in {"=", ":"}): # and (seq2[index-1] in {",", ".", ":"}):
                    logger.debug(f"token: {token1}, {token2} stay after PUNC, and is followed by = or : , it's parameter name, to be static part") #16 only seq2 judge
                    return 0.0, 0
                if index==0 and (seq2[index+1] in {"=", ":"} or re.match(r"^[^:]+::[^:]+$", token1) or re.match(r"^[^:]+::[^:]+$", token2)):
                    logger.debug(f"token: {token1}, {token2} is the first token, and is followed by = or : , it's parameter name, or it's function name, to be static part") #1797
                    return 0.0, 0
                if index>0 and index<len(seq1)-1 and (seq1[index-1] in {"=", ":"} or seq2[index-1] in {"=", ":"}) and (seq1[index+1] in {",", ".", ":"} or seq2[index+1] in {",", ".", ":"}) \
                    and (not nlp(token1)[0].is_oov and not nlp(token2)[0].is_oov):
                    logger.debug(f"token: {token1}, {token2} stay after : , and is followed by PUNC, it's parameter value, to be static part only when it's not OOV")#81 true&false&success
                    return 0.0, 0
                if index>0 and index<len(seq1)-1 and (seq1[index-1] in {":"} or seq2[index-1] in {":"}) and not (seq1[index+1] in {",", ".", ":"} or seq2[index+1] in {",", ".", ":"}) \
                    and (isAllAlphaCapital(token1) and isAllAlphaCapital(token2)):
                    logger.debug(f"token: {token1}, {token2} stay after : , and is not followed by PUNC, it's first token of sentense, to be static part only when all Alpa are capital")#44 IPV4
                    return 0.0, 0
                if index==len(seq1)-1 and index>0 and (seq1[index-1] in {":"} or seq2[index-1] in {":"}) and isAllAlphaCapital(token1) and isAllAlphaCapital(token2):
                    logger.debug(f"token: {token1}, {token2} is the last token and all captial, to be static part") #3 JOB_SETUP
                    return 0.0, 0
                if isAllAlphaCapital(token1) or isAllAlphaCapital(token2):
                    logger.debug(f"token: {token1}, {token2} is all captial, to be static part")
                    return 0.0, 0
                if self.pos_support:
                    lowercaseTokens = [element.lower() for element in seq2]
                    posTagger = nltk.pos_tag(lowercaseTokens)
                    destTokenPosTag = [pos for token, pos in posTagger]
                    logger.debug("destTokenPosTag: %s", destTokenPosTag)

                    if is_determinated_static_part(tokenPosTag[index]) or is_determinated_static_part(destTokenPosTag[index]):
                      logger.debug(f"static part is matched: {token1}: POS type: {tokenPosTag[index]} or {token2}: POS type: {destTokenPosTag[index]}")
                      return 0.0, 0
                    if tokenPosTag[index][0:1] != destTokenPosTag[index][0:1]:
                      logger.debug(f"token: {token1}, {token2} has different POS type: {tokenPosTag[index]} and {destTokenPosTag[index]}")
                      return 0.0, 0
                    elif("VB" in tokenPosTag[index] or "JJ" in tokenPosTag[index] or "NN" in tokenPosTag[index] or \
                          "VB" in destTokenPosTag[index] or "JJ" in destTokenPosTag[index] or "NN" in destTokenPosTag[index]):
                        if (index == 0 and (not tokenPosTag[index+1].startswith("VB") and not destTokenPosTag[index+1].startswith("VB")) \
                            and (not any(char in seq1[index] for char in ['_', '/']) and not any(char in seq2[index] for char in ['_', '/']))):
                            logger.debug(f"token: {token1}, {token2} is the first token and not followed by VB, not special string, to be static part")#1790
                            return 0.0, 0
                        if index>0 and (seq1[index-1] != ":" and seq1[index-1] != "=" and tokenPosTag[index-1] != "IN" and tokenPosTag[index-1] != "TO") \
                            and (seq2[index-1] != ":" and seq2[index-1] != "=" and destTokenPosTag[index-1] != "IN" and destTokenPosTag[index-1] != "TO") \
                            and ((not any(char in seq1[index] for char in ['_', '/']) and not any(char in seq2[index] for char in ['_', '/']))\
                            and (not self.has_numbers(token1) or not self.has_numbers(token2))) \
                            and not (seq1[index-1] == seq2[index-1] and (tokenPosTag[index-1].startswith("JJ") or destTokenPosTag[index-1].startswith("JJ"))):
                            logger.debug(f"token: {token1}, {token2} don't stay after IN or TO type, and not special string and not both having NUM, to be static part")#3215
                            return 0.0, 0
                        if index>0 and "VB" in tokenPosTag[index] and (tokenPosTag[index-1] == "IN" or tokenPosTag[index-1] == "TO"):
                            logger.debug(f"token: {token1}, {token2} is a verb after IN or TO, to be static part")#17 'to spawn'
                            return 0.0, 0
                        if index>0 and "NN" in destTokenPosTag[index] and "VB" in destTokenPosTag[index-1]:
                            logger.debug(f"token: {token1}, {token2} is a noun after verb, to be static part")# removing xxx
                            return 0.0, 0

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
                   include_params: bool) -> tuple[Optional[Template], float, float]:
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

        logger.debug("need to compare to following templateIds: %s", templateIds)
        destTokenOov = []#[nlp(token)[0].is_oov for token in tokens]

        if sim_th == 0.0:
            sim_th = 1 - (math.log(len(tokens), 2) / len(tokens))
            #sim_th = int(sim_th * 10) / 10
            sim_th = round(sim_th, 1)
            logger.debug(f"dynamic sim_th: {sim_th} is calculated.")

        for id in templateIds:
            # Try to retrieve cluster from cache with bypassing eviction
            # algorithm as we are only testing candidates for a match.
            cluster = self.idToTemplateCluster.get(id)
            if cluster is None:
                continue
            cur_sim, param_count = self.get_seq_distance(cluster.getTemplateTokens(), tokens, cluster.getPreemptedTokenSet(), cluster.getTokenPosTag(), include_params)
            logger.debug(f"templateID: {id}, sim: {cur_sim}, sim_th: {sim_th}")
            if cur_sim > max_sim or (cur_sim == max_sim and param_count > max_param_count):
                max_sim = cur_sim
                max_param_count = param_count
                max_cluster = cluster

        if max_sim >= sim_th:
            match_cluster = max_cluster
            logger.debug(f"max_sim:, {max_sim}, sim_th:, {sim_th}")
            logger.debug("fast_match: tree template: %s", match_cluster.getTemplateTokens())
            logger.debug("fast_match: input tokens: %s", tokens)
        return match_cluster, max_sim, max_param_count

    def findMatchedTemplateFromPool(self, length, tokens: Sequence[str]) -> tuple[Template]:
        if length not in self.lengthToTemplateIds:
            return None
        templateIds = self.lengthToTemplateIds.get(length)
        logger.debug("length: {}, size is: {}".format(length, len(templateIds)))
        matchedTemplate, _, _ = self.fast_match(templateIds, tokens, self.sim_th, True)
        return matchedTemplate

    def updateTemplateOfPool(self, template: Template, newTemplateTokens: Sequence[str]) -> None:
        template.setTemplate(newTemplateTokens)
        template.increaseMatchedLogSize()
        # lst = self.lengthToTemplateIds[len(newTemplateTokens)]
        # if template.templateId in lst:
        #     lst.append(lst.pop(lst.index(template.templateId)))

    def findMatchedTemplateFromTree(self,
                    root_node: Node,
                    tokens: Sequence[str],
                    sim_th: float,
                    include_params: bool) -> tuple[Optional[Template], float]:
        result = self.tree_search(root_node, tokens, SequenceType.FORWARD, sim_th, include_params)
        if result is not None:
            fw, fw_sim, fw_paraCnt = result
        else:
            fw, fw_sim, fw_paraCnt = None, 0.0, 0.0
        if self.bi_tree_support:
            result =  self.tree_search(root_node, tokens, SequenceType.REVERSE, sim_th, include_params)
            if result is not None:
                rv, rv_sim, rv_paraCnt = result
            else:
                rv, rv_sim, rv_paraCnt = None, 0.0, 0.0
        else:
            rv, rv_sim, rv_paraCnt = None, 0.0, 0.0

        return (fw, fw_sim, fw_paraCnt), (rv, rv_sim, rv_paraCnt)

    def tree_search(self,
                    root_node: Node,
                    tokens: Sequence[str],
                    seq_type: SequenceType,
                    sim_th: float,
                    include_params: bool) -> tuple[Optional[Template], float, float]:

        # at first level, children are grouped by token (word) count
        token_count = len(tokens)
        logger.debug("token_count: %s", token_count)
        cur_node = root_node.keyToChildNode.get(str(token_count))
        logger.debug("length node: %s", cur_node)
        # no template with same token count yet
        if cur_node is None:
            return None, 0.0, 0.0

        cur_node = cur_node.keyToChildNode.get(str(int(seq_type.value)))
        logger.debug("direction node: %s", cur_node)
        # no template with matched dirction sequence yet
        if cur_node is None:
            return None, 0.0, 0.0
        logger.debug("cur_node: %s", cur_node)
        # handle case of empty log string - return the single cluster in that group
        if token_count == 0:
            return self.idToTemplateCluster.get(cur_node.templateIds[0]), 0.0, 0.0

        # find the leaf node for this log - a path of nodes matching the first N tokens (N=tree depth)
        tokensToMatch = tokens[::1] if seq_type == SequenceType.FORWARD else tokens[::-1]
        logger.debug("tokens: %s", tokens)

        for token in tokensToMatch:
            logger.debug("start to match token: %s", token)
            if cur_node.nodeType == NodeType.LEAF:
                logger.debug("leaf node is matched")
                break
            keyToChildNode = cur_node.keyToChildNode
            token_node = keyToChildNode.get(token)
            wildcard_node = keyToChildNode.get(self.param_str)
            if token_node is not None and wildcard_node is not None: # checke whether <*> has preempted token
                logger.debug("match both token and <*>, token is: {}, tokensInWildcard: {}".format(token, wildcard_node.tokensInWildcard))
                if token in wildcard_node.tokensInWildcard:
                    cur_node = wildcard_node
                else:
                    cur_node = token_node
            elif token_node is not None:
                cur_node = token_node
            elif wildcard_node is not None:
                cur_node = wildcard_node
            else:
                return None, 0.0, 0.0
            # token is matched:
            logger.debug("branch match:" f'{token} is matched')

        # get best match among all clusters with same prefix, or None if no match is above sim_th
        logger.debug("branch is matched, go to fast_match")
        cluster, sim, paraCnt = self.fast_match(cur_node.templateIds, tokens, sim_th, include_params)
        return cluster, sim, paraCnt

    def buildTemplateWithInputLog(self, length, tokens: Sequence[str]) -> Optional[Template]:
        template = Template(tokens, self.getNewTemplateId(), isPosSupported=self.pos_support)
        self.idToTemplateCluster[template.templateId] = template
        #self.lengthToTemplateIds[length].append(template.templateId)
        self.lengthToTemplateIds[length].insert(0, template.templateId)
        logger.debug("length: {}, str is: {}".format(length, tokens))
        return template

    def addTemplateSeqToPrefixTree(self, root_node: Node, template: Template) -> None:
        self.add_seq_to_prefix_tree(root_node, template, SequenceType.FORWARD)
        if self.bi_tree_support:
            self.add_seq_to_prefix_tree(root_node, template, SequenceType.REVERSE)

    def add_log_message(self, content: str) -> Tuple[Template, str]:
        content_tokens = self.get_content_as_tokens(content)
        length = len(content_tokens)
        logger.debug("========input log is============: %s",content_tokens)
        if self.profiler:
            self.profiler.start_section("findMatchedTemplateFromTree")
        (fwSeqMatchedTemplate, fwSim, fwParaCnt), (RvSeqMatchedTemplate, rvSim, rvParaCnt) = self.findMatchedTemplateFromTree \
                                                                (self.root_node, content_tokens, self.sim_th, include_params=True)
        logger.debug(f"tree_search return is:({fwSeqMatchedTemplate}, {RvSeqMatchedTemplate})")
        if self.profiler:
            self.profiler.end_section()

        # Match no existing template
        if fwSeqMatchedTemplate is None and RvSeqMatchedTemplate is None: # both forward and reverse sequence don't have matched template
            if self.profiler:
                self.profiler.start_section("Match no existing template, findMatchedTemplateFromPool")



            #matchedPoolTemplate = self.findMatchedTemplateFromPool(length, content_tokens)
            matchedPoolTemplate = None

            if self.profiler:
                self.profiler.end_section()
            if matchedPoolTemplate is None: # it's a new message which don't have template in pool yet
                matchedPoolTemplate = self.buildTemplateWithInputLog(length, content_tokens)
                update_type = "created"
            else: # similar template is found in pool but not in tree, need update template and add to tree
                newTemplateTokens = self.create_template(content_tokens, matchedPoolTemplate.getTemplateTokens())
                self.updateTemplateOfPool(matchedPoolTemplate, newTemplateTokens)
                update_type = "updated"
            self.addTemplateSeqToPrefixTree(self.root_node, matchedPoolTemplate)


        else: # Match existing template at least one direction of tree
            if self.profiler:
                self.profiler.start_section("Match any existing template")

            if fwSeqMatchedTemplate is not None and RvSeqMatchedTemplate is not None:
                if(fwSeqMatchedTemplate.templateId != RvSeqMatchedTemplate.templateId):
                    logger.debug("input is: %s",content_tokens)
                    logger.debug("fwSeqMatchedTemplate.templateId: %s", fwSeqMatchedTemplate.templateId)
                    logger.debug("fw template: %s", fwSeqMatchedTemplate.getTemplateStr())
                    logger.debug("RvSeqMatchedTemplate.templateId: %s", RvSeqMatchedTemplate.templateId)
                    logger.debug("Rv template: %s", RvSeqMatchedTemplate.getTemplateStr())
                    #matchedPoolTemplate = fwSeqMatchedTemplate if fwSim > rvSim else RvSeqMatchedTemplate
                    if fwSim > rvSim:
                        matchedPoolTemplate = fwSeqMatchedTemplate
                    elif fwSim < rvSim:
                        matchedPoolTemplate = RvSeqMatchedTemplate
                    elif fwSim == rvSim:
                        if fwParaCnt <= rvParaCnt:
                            matchedPoolTemplate = fwSeqMatchedTemplate
                        else:
                            matchedPoolTemplate = RvSeqMatchedTemplate
                    assert(1)
                else:
                    matchedPoolTemplate = fwSeqMatchedTemplate
                #assert(fwSeqMatchedTemplate.templateId == RvSeqMatchedTemplate.templateId)
            elif fwSeqMatchedTemplate is not None:
                matchedPoolTemplate = fwSeqMatchedTemplate
            else:
                matchedPoolTemplate = RvSeqMatchedTemplate

            newTemplateTokens = self.create_template(content_tokens, matchedPoolTemplate.getTemplateTokens())
            if newTemplateTokens != matchedPoolTemplate.getTemplateTokens():
                update_type = "updated"
                self.updateTemplateOfPool(matchedPoolTemplate, newTemplateTokens)
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

