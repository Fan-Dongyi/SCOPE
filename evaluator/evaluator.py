import datetime
import pandas as pd
import os, re
import json
import logging
import sys
import time
from os.path import dirname
import logging
import logging.config
import re
import math

#logging.basicConfig(filename="./result.log", filemode="w", level=logging.INFO, format="%(levelname)s - %(message)s")
#logging.basicConfig(filename="./result.log", filemode="w", level=logging.DEBUG, format="%(levelname)s - %(message)s")

result_logger = logging.getLogger("result_logger")
result_logger.setLevel(logging.INFO)
#result_logger.setLevel(logging.DEBUG)
result_handler = logging.FileHandler("./result.log", mode="w")
result_handler.setFormatter(logging.Formatter("%(levelname)s - %(message)s"))
result_logger.addHandler(result_handler)

BLUE = "\033[34m"
RESET = "\033[0m"
YELLOW = "\033[33m"
GREEN = "\033[32m"


"""benchmark_settings = {

    'Proxifier': {
        'log_file': 'Proxifier/Proxifier_2k.log',
        'log_format': '\\[<Time>\\] <Program> - <Content>',
        'regex': [r'<\d+\ssec', r'([\w-]+\.)+[\w-]+(:\d+)?', r'\d{2}:\d{2}(:\d{2})*', r'[KGTM]B'],
        'delimiter': [r'\(.*?\)'],
        'tag': 0,
        'theshold': 3
    },
    'HDFS': {
        'log_file': 'HDFS/HDFS_2k.log',
        'log_format': '<Date> <Time> <Pid> <Level> <Component>: <Content>',
        'regex': [r'blk_-?\d+', r'(\d+\.){3}\d+(:\d+)?'],
        'delimiter': [''],
        'tag': 0,
        'theshold': 2
        },

    'Hadoop': {
        'log_file': 'Hadoop/Hadoop_2k.log',
        'log_format': '<Date> <Time> <Level> \[<Process>\] <Component>: <Content>',
        'regex': [r'(\d+\.){3}\d+'],
        'delimiter': [],
        'tag': 1,
        'theshold': 6
        },

    'Spark': {
        'log_file': 'Spark/Spark_2k.log',
        'log_format': '<Date> <Time> <Level> <Component>: <Content>',
        'regex': [r'(\d+\.){3}\d+', r'\b[KGTM]?B\b', r'([\w-]+\.){2,}[\w-]+'],
        'delimiter': [],
        'tag': 0,
        'theshold': 4
        },

    'Zookeeper': {
        'log_file': 'Zookeeper/Zookeeper_2k.log',
        'log_format': '<Date> <Time> - <Level>  \[<Node>:<Component>@<Id>\] - <Content>',
        'regex': [r'(/|)(\d+\.){3}\d+(:\d+)?'],
        'delimiter': [],
        'tag': 1,
        'theshold': 3
        },

    'BGL': {
        'log_file': 'BGL/BGL_2k.log',
        'log_format': '<Label> <Timestamp> <Date> <Node> <Time> <NodeRepeat> <Type> <Component> <Level> <Content>',
        'regex': [r'core\.\d+'],
        'delimiter': [],
        'theshold': 6
        },

    'HPC': {
        'log_file': 'HPC/HPC_2k.log',
        'log_format': '<LogId> <Node> <Component> <State> <Time> <Flag> <Content>',
        'regex': [],
        'delimiter': [],
        'theshold': 5
        },

    'Thunderbird': {
        'log_file': 'Thunderbird/Thunderbird_2k.log',
        'log_format': '<Label> <Timestamp> <Date> <User> <Month> <Day> <Time> <Location> <Component>(\[<PID>\])?: <Content>',
        'regex': [r'(\d+\.){3}\d+'],
        'delimiter': [],
        'theshold': 3
        },

    'Windows': {
        'log_file': 'Windows/Windows_2k.log',
        'log_format': '<Date> <Time>, <Level>                  <Component>    <Content>',
        'regex': [r'0x.*?\s'],
        'delimiter': [],
        'theshold': 3
        },

    'Linux': {
        'log_file': 'Linux/Linux_2k.log',
        'log_format': '<Month> <Date> <Time> <Level> <Component>(\[<PID>\])?: <Content>',
        'regex': [r'(\d+\.){3}\d+', r'\d{2}:\d{2}:\d{2}',r'J([a-z]{2})'],
        'delimiter': [r''],
        'theshold': 4
        },

    'Android': {
        'log_file': 'Android/Android_2k.log',
        'log_format': '<Date> <Time>  <Pid>  <Tid> <Level> <Component>: <Content>',
        'regex': [r'(/[\w-]+)+', r'([\w-]+\.){2,}[\w-]+', r'\b(\-?\+?\d+)\b|\b0[Xx][a-fA-F\d]+\b|\b[a-fA-F\d]{4,}\b'],
        'delimiter': [r''],
        'theshold': 5
        },

    'HealthApp': {
        'log_file': 'HealthApp/HealthApp_2k.log',
        'log_format': '<Time>\|<Component>\|<Pid>\|<Content>',
        'regex': [],
        'delimiter': [r''],
        'theshold': 4
        },

    'Apache': {
        'log_file': 'Apache/Apache_2k.log',
        'log_format': '\[<Time>\] \[<Level>\] <Content>',
        'regex': [r'(\d+\.){3}\d+'],
        'delimiter': [],
        'theshold': 4
        },

    'OpenSSH': {
        'log_file': 'OpenSSH/OpenSSH_2k.log',
        'log_format': '<Date> <Day> <Time> <Component> sshd\[<Pid>\]: <Content>',
        'regex': [r'(\d+\.){3}\d+', r'([\w-]+\.){2,}[\w-]+'],
        'delimiter': [],
        'theshold': 6
        },

    'OpenStack': {
        'log_file': 'OpenStack/OpenStack_2k.log',
        'log_format': '<Logrecord> <Date> <Time> <Pid> <Level> <Component> \[<ADDR>\] <Content>',
        'regex': [r'((\d+\.){3}\d+,?)+', r'/.+?\s ', r'\d+'],
        'delimiter': [],
        'theshold': 5,
        },

    'Mac': {
        'log_file': 'Mac/Mac_2k.log',
        'log_format': '<Month>  <Date> <Time> <User> <Component>\[<PID>\]( \(<Address>\))?: <Content>',
        'regex': [r'([\w-]+\.){2,}[\w-]+'],
        'delimiter': [],
        'theshold': 5
        },
} """

benchmark_settings1 = {



    'Mac': {
        'log_file': 'Mac/Mac_2k.log',
        'log_format': '<Month>  <Date> <Time> <User> <Component>\\[<PID>\\]( \\(<Address>\\))?: <Content>',
        #'regex': [r'([\w-]+\.){2,}[\w-]+'],
        'regex': [],
        'index_list': [0, 1, 2, 3, 4],
        'filter': [],
        'st': 0.6,
        'depth': 7
        },


    'Linux': {
        'log_file': 'Linux_corrected/Linux_2k.log',
        'log_format': '<Month> <Date> <Time> <Level> <Component>(\\[<PID>\\])?: <Content>',
        #'regex': [r'(\d+\.){3}\d+', r'\d{2}:\d{2}:\d{2}'],
        'regex': [],
        'filter': [],
        'index_list': [0, ],
        'st': 0.1,
        'depth': 5,
        },
    'HealthApp': {
        'log_file': 'HealthApp/HealthApp_2k.log',
        'log_format': '<Time>\\|<Component>\\|<Pid>\\|<Content>',
        'regex': [],
        'filter': [],
        'index_list': [0, 1, 2, 3, 4],
        'st': 0.4,
        'depth': 8
        },

}

benchmark_settings = {
    'HPC': {
        'log_file': 'HPC/HPC_2k.log',
        'log_format': '<LogId> <Node> <Component> <State> <Time> <Flag> <Content>',
        'regex': [],
        'filter': []
        },

    'OpenStack': {
        'log_file': 'OpenStack/OpenStack_2k.log',
        'log_format': '<Logrecord> <Date> <Time> <Pid> <Level> <Component> \\[<ADDR>\\] <Content>',
        'regex': ["(\\w+-\\w+-\\w+-\\w+-\\w+)", r'HTTP\/\d+\.\d+'],
        'filter': [r'HTTP\/\d+\.\d+', ]
        },

    'BGL': {
        'log_file': 'BGL/BGL_2k.log',
        'log_format': '<Label> <Timestamp> <Date> <Node> <Time> <NodeRepeat> <Type> <Component> <Level> <Content>',
        'regex': [],
        'filter': []
        },

    'HDFS': {
        'log_file': 'HDFS/HDFS_2k.log',
        'log_format': '<Date> <Time> <Pid> <Level> <Component>: <Content>',
        'regex': [r'blk_-?\d+'],
        'filter': []
        },

    'Hadoop': {
        'log_file': 'Hadoop/Hadoop_2k.log',
        'log_format': '<Date> <Time> <Level> \\[<Process>\\] <Component>: <Content>',
        'regex': [r'\[.*?(_.*?)+\]', ],
        'filter': []
        },

    'Spark': {
        'log_file': 'Spark/Spark_2k.log',
        'log_format': '<Date> <Time> <Level> <Component>: <Content>',
        'regex': [],
        'filter': []
        },

    'Zookeeper': {
        'log_file': 'Zookeeper/Zookeeper_2k.log',
        'log_format': '<Date> <Time> - <Level>  \\[<Node>:<Component>@<Id>\\] - <Content>',
        'regex': [],
        'filter': []
        },

    'Thunderbird': {
        'log_file': 'Thunderbird/Thunderbird_2k.log',
        'log_format': '<Label> <Timestamp> <Date> <User> <Month> <Day> <Time> <Location> <Component>(\\[<PID>\\])?: <Content>',
        'regex': [r'(\d+\.){3}\d+'],
        'filter': []
        },

    'Windows': {
        'log_file': 'Windows/Windows_2k.log',
        'log_format': '<Date> <Time>, <Level>                  <Component>    <Content>',
        'regex': [],
        'filter': []
        },

    'Linux': {
        'log_file': 'Linux_corrected/Linux_2k.log',
        'log_format': '<Month> <Date> <Time> <Level> <Component>(\\[<PID>\\])?: <Content>',
        'regex': [],
        'filter': []
        },

    'Andriod': {
        'log_file': 'Andriod/Andriod_2k.log',
        'log_format': '<Date> <Time>  <Pid>  <Tid> <Level> <Component>: <Content>',
        'regex': [r'(/[\w-]+)+', r'([\w-]+\.){2,}[\w-]+', r'\b(\-?\+?\d+)\b|\b0[Xx][a-fA-F\d]+\b|\b[a-fA-F\d]{4,}\b',
                  r'-\<\*\>'],
        'filter': []
        },

    'HealthApp': {
        'log_file': 'HealthApp/HealthApp_2k.log',
        'log_format': '<Time>\\|<Component>\\|<Pid>\\|<Content>',
        'regex': [],
        'filter': []
        },

    'Apache': {
        'log_file': 'Apache/Apache_2k.log',
        'log_format': '\\[<Time>\\] \\[<Level>\\] <Content>',
        'regex': [r'\/(?:\w+\/){2,}\w+\.\w+$'],
        'filter': []
        },

    'Proxifier': {
        'log_file': 'Proxifier/Proxifier_2k.log',
        'log_format': '\\[<Time>\\] <Program> - <Content>',
        'regex': [r'<\d+\ssec', r'([\w-]+\.)+[\w-]+(:\d+)?', r'\d{2}:\d{2}(:\d{2})*', r'[KGTM]B'],
        "filter": [r' \(\d+(\.\d+)?\s(?:K|M)B\)', ]
        },

    'OpenSSH': {
        'log_file': 'OpenSSH/OpenSSH_2k.log',
        'log_format': '<Date> <Day> <Time> <Component> sshd\\[<Pid>\\]: <Content>',
        'regex': [r"(\d+):"],
        'filter': []
        },

    'Mac': {
        'log_file': 'Mac/Mac_2k.log',
        'log_format': '<Month>  <Date> <Time> <User> <Component>\\[<PID>\\]( \\(<Address>\\))?: <Content>',
        'regex': [],
        'filter': []
        },

    'Thunderbird_expand': {
        'log_file': 'Thunderbird_expand/Thunderbird_2k.log',
        'log_format': '<Label> <Timestamp> <Date> <User> <Month> <Day> <Time> <Location> <Component>(\\[<PID>\\])?: <Content>',
        'regex': [],
        'filter': []
        },
}

class format_log:    # this part of code is from LogPai https://github.com/LogPai
    def __init__(self, log_format, indir='./'):
        self.path = indir
        self.logName = None
        self.df_log = None
        self.log_format = log_format

    def get_format_logs(self, logName):
        self.logName=logName
        self.load_data()
        return self.df_log

    def generate_logformat_regex(self, logformat):
        """ Function to generate regular expression to split log messages
        """
        headers = []
        splitters = re.split(r'(<[^<>]+>)', logformat)
        regex = ''
        for k in range(len(splitters)):
            if k % 2 == 0:
                splitter = re.sub(r' +', r'\\s+', splitters[k])
                regex += splitter
            else:
                header = splitters[k].strip('<').strip('>')
                regex += '(?P<%s>.*?)' % header
                headers.append(header)
        regex = re.compile('^' + regex + '$')
        return headers, regex

    def log_to_dataframe(self, log_file, regex, headers, logformat):
        """ Function to transform log file to dataframe
        """
        log_messages = []
        linecount = 0
        with open(log_file, 'r', encoding='UTF-8') as fin:
            for line in fin.readlines():
                try:
                    match = regex.search(line.strip())
                    message = [match.group(header) for header in headers]
                    log_messages.append(message)
                    linecount += 1
                except Exception as e:
                    pass
        logdf = pd.DataFrame(log_messages, columns=headers)
        logdf.insert(0, 'LineId', None)
        logdf['LineId'] = [i + 1 for i in range(linecount)]
        return logdf


    def load_data(self):
        headers, regex = self.generate_logformat_regex(self.log_format)
        self.df_log = self.log_to_dataframe(os.path.join(self.path, self.logName), regex, headers, self.log_format)


def preprocess(line, rex, filter):
    for currentFil in filter:
        line = re.sub(currentFil, '', line)
    for currentRex in rex:
        line = re.sub(currentRex, '<*>', line)
    return line

benchmark_result=[]

# Add the directory containing the module to sys.path
scope_dir = os.path.join(os.path.dirname(__file__), "../code")
sys.path.insert(0, scope_dir)  # Insert at the start of sys.path

from template_miner_tools import TemplateMiner
from template_miner_config import TemplateMinerConfig

#in_log_file = os.path.join(os.path.dirname(__file__), "scopeTestFile.txt")
config = TemplateMinerConfig()
config.load(f"{dirname(__file__)}/../examples/tools.ini")
config.profiling_enabled = True
#template_miner = TemplateMiner(config=config)

for dataset, setting in benchmark_settings.items():
    result_logger.debug(BLUE+dataset+RESET)
    starttime = datetime.datetime.now()
    parse = format_log(log_format=setting['log_format'], indir=dirname(__file__)+'/logs/')
    logs = parse.get_format_logs(setting['log_file'])
    content = logs['Content']
    start = datetime.datetime.now()
    sentences = content.tolist()
    #df_groundtruth=pd.read_csv(dirname(__file__) + '/logs/' + dataset + '/' + dataset + '_2k.log_structured.csv',
                #encoding='UTF-8', header=0)
    df_groundtruth = pd.read_csv(os.path.join(dirname(__file__)+'/logs/', setting['log_file'] + '_structured.csv'), encoding="utf-8")
    df_data = pd.DataFrame()

    #config.drain_sim_th = setting['st']
    #config.drain_depth = setting['depth']
    template_miner = TemplateMiner(config=config)

    start_time = time.time()
    batch_start_time = start_time
    batch_size = 10000
    line_count = 0
    log_templateIds = []
    log_templateStrs = []

    for line in sentences:
        line = line.rstrip()
        line = preprocess(line, setting['regex'], setting['filter'])
        result = template_miner.add_log_message(line)
        line_count += 1
        if line_count % batch_size == 0:
            time_took = time.time() - batch_start_time
            rate = batch_size / time_took
            #result_logger.info(f"Processing line: {line_count}, rate {rate:.1f} lines/sec, "                    f"{len(template_miner.drain.clusters)} clusters so far.")
            batch_start_time = time.time()
        if result["change_type"] != None:
            result_json = json.dumps(result)
            #result_logger.info(f"Input ({line_count}): {line}")
            #result_logger.info(f"Result: {result_json}")

        log_templateIds.append(result["cluster_id"])
        log_templateStrs.append(result["template_mined"])

    time_took = time.time() - start_time
    rate = line_count / time_took

    df_data['EventId'] = log_templateIds
    df_data['EventTemplate'] = log_templateStrs

    count = 0
    data = df_data['EventId']
    groundtruth = df_groundtruth['EventId']
    for parsed_eventId in data.value_counts().index:
        logIds = data[data == parsed_eventId].index
        result_logger.debug("\n********output lines for template id:", parsed_eventId, logIds)
        result_logger.debug("********output original lines for template:", df_data['EventTemplate'][logIds[0]]) # the first line to generate template = original log
        result_logger.debug("********output final lines for template:", df_data['EventTemplate'][logIds[-1]]) # the last line to update template = final template
        series_groundtruth_logId_valuecounts = groundtruth[logIds].value_counts()
        result_logger.debug("====label situation for above lines", series_groundtruth_logId_valuecounts)
        if series_groundtruth_logId_valuecounts.size == 1:
            groundtruth_eventId = series_groundtruth_logId_valuecounts.index[0]
            x = groundtruth[groundtruth == groundtruth_eventId]
            if logIds.size == groundtruth[groundtruth == groundtruth_eventId].size:
                count += logIds.size
                result_logger.debug("&&&&&& count: ", count)
            else:
                result_logger.debug("groundtruth lines: ", x.index.tolist())
                result_logger.debug("groundtruth lD: ", groundtruth_eventId)
                result_logger.debug("groundtruth log:", df_groundtruth['Content'][x.index[0]])
                result_logger.debug("groundtruth template:", df_groundtruth['EventTemplate'][x.index[0]])
                result_logger.debug("groundtruth size: ", x.size)
                result_logger.debug("data size: ", len(logIds))
                diff = x.index.difference(logIds)
                result_logger.debug("different lines: ", diff)
                result_logger.debug("wrong judged template:", df_data['EventTemplate'][diff[0]])
                result_logger.debug("wrong judged template:", df_data['EventTemplate'][diff[-1]])
        else:
            result_logger.debug("########label:", df_groundtruth['EventTemplate'][logIds])
            result_logger.debug("@@@@@output:", df_data['EventTemplate'][logIds])
    accuracy = round(float(count) / data.size, 4)
    accuracy = f"{accuracy:.4f}"
    result_logger.debug('\n=== Evaluation on %s ==='%dataset)
    result_logger.debug(accuracy)

    result_logger.info(f"Evaluation on Dataset: {dataset}, Group Accuracy: {accuracy}, Duration: {time_took:.2f} sec, "
                f"Total of {line_count} lines, rate {rate:.1f} lines/sec, {len(template_miner.scope.clusters)} clusters")

import re
import pandas as pd
from tabulate import tabulate

def parse_file_to_table(file_path):
    # 定义正则表达式解析数据
    pattern = re.compile(
        r"Evaluation on Dataset: (\S+), Group Accuracy: ([\d.]+), Duration: ([\d.]+) sec, Total of (\d+) lines, rate ([\d.]+) lines/sec, (\d+) clusters"
    )
    # 存储提取的数据
    data = []

    # 读取文件并解析
    with open(file_path, "r", encoding="utf-8") as file:
        for line in file:
            match = pattern.search(line)
            if match:
                dataset, accuracy, duration, line_number, rate, clusters = match.groups()
                data.append([dataset, float(accuracy), float(duration), int(line_number), float(rate), int(clusters)])

    # 使用 Pandas 创建表格
    df = pd.DataFrame(data, columns=["Dataset", "Group Accuracy", "Duration (sec)", "Line Number", "Rate (lines/sec)", "Clusters"])
    return df

# 文件路径
file_path = "./result.log"

# 调用函数并打印表格
table = parse_file_to_table(file_path)
print(tabulate(table, headers="keys", tablefmt="grid"))

# 如果需要保存为 CSV 文件
table.to_csv("output.csv", index=False)


#     df_output,template_set= Brain.parse(sentences, setting['regex'], dataset, setting['theshold'], setting['delimiter'], starttime, efficiency=False, df_input=df_groundtruth.copy())
#     Brain.save_result(dataset, df_output, template_set)
#     f_measure, accuracy= evaluator.evaluate(df_groundtruth, df_output)
#     GA= evaluator.get_GA(df_groundtruth, df_output)
#     ED,ED_= evaluator.get_editdistance(df_groundtruth, df_output)
#     benchmark_result.append([dataset, GA, f_measure,ED])
# print('\n=== Overall evaluation results ===')
# pd.set_option('display.max_columns', None)
# pd.set_option('display.max_colwidth', None)
# df_result = pd.DataFrame(benchmark_result, columns=['Dataset',  'Group_accuracy', 'F1_score','Edit_distance'])
# df_result.set_index('Dataset', inplace=True)
# print(GREEN)
# print(df_result)
# print(RESET)
# print("Average Group_accuracy= "+YELLOW+str(sum(df_result['Group_accuracy'])/len(df_result['Group_accuracy']))+RESET+ \
# "   Average Edit_distance (without data clean) = "+YELLOW+str(sum(df_result['Edit_distance'])/len(df_result['Edit_distance']))+RESET)




