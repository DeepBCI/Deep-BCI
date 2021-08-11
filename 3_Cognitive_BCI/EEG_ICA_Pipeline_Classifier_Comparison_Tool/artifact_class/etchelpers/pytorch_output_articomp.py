## Output file handler for pytorch training in artifact comparisons


import scipy
from scipy.stats import stats as scistat
import numpy as np
import pickle
import torch
import os, sys
from .simulate_chance import simulate_chance


# class BaseMetric:
#     def __init__(self, metric_name, value):
#         self.name = metric_name
#         self.value = value
# keep metrics in dictionary forms



class MetricAggr:
    def __init__(self, metric_name: str, metric_members, group_type, member_type):
        self.metric_name = metric_name

        self.group_type = group_type
        self.member_type = member_type

        self.mvalues =0

    def mean(self):
        return np.mean(self.mvalues)

    def std(self):
        return np.std(self.mvalues)

    def median(self):
        return np.median(self.mvalues)

    def mode(self):
        return scipy.mode(self.mvalues)

    def values_as_arr(self):
        return self.mvalues

    def values_as_dict(self):
        return


class MetricContainer:
    # contains different metrics for a unit model observation
    # in standard neural nets this would represent result from one epoch
    # in standard ldas this would represent result from one fold

    def __init__(self, container_type=None, container_info=None):
        self.container_type = container_type
        self.metrics = {}
        self.metric_names = []
        self.container_info = container_info
        if container_info is None:
            self.container_info = {}

    @classmethod
    def with_metrics(cls, metrics, container_info=None):
        cls.metrics = {}
        cls.metric_names = []
        cls.container_info = container_info
        if container_info is None:
            cls.container_info = {}

        cls.add_metrics(metrics)

    def add_metrics(self, metric):
        for key, item in metric.items():
            self.metrics[key] = item

        # ???? self.metrics should always be a dictionary
        # what the hell were you on when you wrote this
        # # insert multiple metrics
        # if type(metric) is list:
        #     self.metrics += metric
        #     self.metric_names += [mt['name'] for mt in metric]
        # # single metric insertion
        # elif type(metric) is dict:
        #     self.metrics.append(metric)
        #     self.metric_names.append(metric['name'])

    def add_container_info(self, infod):
        for key, item in infod.items():
            self.container_info[key] = item

    def get_metric_names(self):
        return [metname for metname, met in self.metrics.items()]


class MetricAggregates(MetricContainer):
    # contains MetricContainers or Lower MetricAggregates as members

    def __init__(self, container_type=None,  container_info=None, member_type=None):
        super().__init__(container_type=container_type, container_info=container_info)

        self.members = []
        self.member_type = member_type
        self.members_n = 0


        #self.agglevel = 0
        #self.agglevel_toggle = True
    @classmethod
    def with_members(cls, members, member_type='individual', container_type=None, container_info=None):
        super().__init__(container_type=container_type, container_info=container_info)

        cls.members = []
        cls.members_n = 0
        cls.member_type = member_type

        # cls.agglevel = 0
        # cls.agglevel_toggle = True

        # call this last
        cls.add_members(members)

        return cls

    def add_members(self, member, member_type='individual'):
        # at this point it doesn't care whether members are more aggregates or a MetricContainer
        if type(member) is list:
            self.members += member
            self.members_n = len(self.members)
        if type(member) is MetricAggregates:
            self.members.append(member)
            self.members_n += 1
        if type(member) is MetricContainer:
            self.members.append(member)
            self.members_n += 1
        # and set member type if it's not set yet
        if self.member_type is None:
            self.member_type = member_type

    def member_statistics(self, exclusion_rules=None):
        if exclusion_rules is None:
            exclusion_rules = ['_loss', '_pred', '_target', 'std_', 'median_', 'sem_']
        for metric in self.members[0].metrics.keys():
            if any(substring in metric for substring in exclusion_rules):
                continue
            #if '_pred' in metric or '_target' in metric:
                #continue
            member_mets = [mb.metrics[metric] for mb in self.members]
            # stats to run: mean, median, mode, std, max, min
            self.metrics[metric] = np.mean(member_mets)
            self.metrics['median_' + metric] = np.median(member_mets)
            self.metrics['std_' + metric] = np.std(member_mets)
            self.metrics['sem_' + metric] = scistat.sem(member_mets)

        return

    def apply_stats_on_each_member(self, exclusion_rules=None):
        for member in self.members:
            member.member_statistics(exclusion_rules=exclusion_rules)

        return

    def extract_metric_from_members(self, metric):
        ext_met = [mem.metrics[metric] for mem in self.members]
        return ext_met




    # do we assume that members at the same level have the same set of metrics, all the way down?
    # yes, it should be that way in an identical run
    # def extract_member_metrics(self):
    #
    #     # would member aggregate's "metrics" be the average stuff?
    #
    #     # for each metric in each member, create a aggregate representative list of dicts..
    #     # or dict of lists. simply return it
    #     for member in self.members:
    #         #but depending on the level the expected metric may be in fact different
    #         # e.g. if members are epos then agg may look for last/highest etc.
    #         # on the other hand if members are folds we may look for
    #         # or if members are participants we may look for fold averages
    #
    #         # if member is a container
    #         # if member is an aggregate
    #         if member is MetricAggregates:
    #             for metric in member.members:
    #
    #         elif member is MetricContainer:
    #
    #             for metric in member.metrics:
    #
    #     self.representative
    #     return
    #
    # def get_average_metrics(self):
    #     return

    # def count_agglevels(self):
    #
    #     if self.agglevel_toggle:
    #         if not self.members:
    #             self.agglevel = 0
    #         elif self.members[0] is MetricAggregates:
    #             self.agglevel += 1
    #             self.members[0].count_agglevels()
    #         elif self.members[0] is MetricContainer:
    #             self.agglevel += 1
    #         else:
    #             return
    #     return
    #
    # def use_agglevel_counter(self):



def simchance_on_members(metric_agg, metric_to_test, outname_prefix, alpha=0.05, nsims=25000):
    '''
    Assumes that self.members contains metrics on individuals,
    because that is the only way you'd be able to run these
    :param metric:
    :param metric_to_test: name of metric to test e.g. ['bets_epo_test_acc', '..']
    :param outname_prefix: name of metric tested e.g. ['bets_epo_test_acc_beyond', '..']
    :return:
    '''
    if 'output_classn' in metric_agg.container_info['dataset_config']:
        nclass_val = metric_agg.container_info['dataset_config']['output_classn']
    else:
        nclass_val = 2

    for member_idx, member in enumerate(metric_agg.members):
        simacc = simulate_chance(member.container_info['n_after_bal'], alpha=alpha,
                                 nclass=nclass_val, nsims=nsims)
        metric_agg.members[member_idx].add_metrics({'simulated_chance': simacc})

        for metrictt_idx, metrictt in enumerate(metric_to_test):
            metric_agg.members[member_idx].add_metrics({'{}_beyond_ci'.format(outname_prefix[metrictt_idx]):
                                                        metric_agg.members[member_idx].metrics[metrictt] > simacc[2]})

    return metric_agg

def simchance_on_itself(metric_agg, metric_to_test, outname_prefix, alpha=0.05, nsims=25000):
    if 'output_classn' in metric_agg.container_info['dataset_config']:
        nclass_val = metric_agg.container_info['dataset_config']['output_classn']
    else:
        nclass_val = 2


    simacc = simulate_chance(metric_agg.container_info['n_after_bal'], alpha=alpha,
                             nclass=nclass_val, nsims=nsims)
    metric_agg.add_metrics({'simulated_chance': simacc})

    for metrictt_idx, metrictt in enumerate(metric_to_test):
        metric_agg.add_metrics({'{}_beyond_ci'.format(outname_prefix[metrictt_idx]):
                                                        metric_agg.metrics[metrictt] > simacc[2]})

    return metric_agg


def average_conf_matrix(metagg: MetricAggregates, fconf_mat_name):
    member_confs = []
    # first get confmusion mat of each participant
    for member in metagg.members:
        member_mets = [mb.metrics[fconf_mat_name] for mb in member.members]
        member_conf = np.mean(member_mets, axis=0)
        member_confs.append(member_conf)
        member.metrics[fconf_mat_name] = member_conf

    return metagg


class MsetComparator():
    def __init__(self, met_ica: MetricAggregates, met_noica: MetricAggregates):
        self.met_ica = met_ica
        self.met_noica = met_noica

        self.comp_metrics = {}
        self.task_info = {}

    def add_metric(self, metric:dict):
        for metkey, met in metric.items():
            self.comp_metrics[metkey] = met

    def add_taskinfo(self, taskinfo:dict):
        for metkey, met in taskinfo.items():
            self.task_info[metkey] = met


class SetRunViewer():
    '''
    collator of a single dataset's different runsets under one algorithm.
    constructs dictionary of metric matrices, each item corresponding to one metric
    under this assumption,
    an indi metrics matrix would contain metrics(e.g.test accuracies) in:
    run(preproc_type) x participant\
    a group matrix would contain run(preproc_type)
    Extended to AlgowideSetViewer
    '''
    def __init__(self, runsets, dsetname):
        self.runsets = runsets
        self.indi_metrics_matrices = {}
        self.group_metrics_matrices = {}
        self.row_order = [rs.container_type for rs in self.runsets]
        self.dsetname = dsetname

    def construct_indi_metric_matrix(self, metric_names):
        for metricidx, metric in enumerate(metric_names):
            indi_metric_matrix = []

            for rs_idx, rs in enumerate(self.runsets):
                matrix_row = [sbj.metrics[metric] for sbj in rs.members]
                indi_metric_matrix.append(matrix_row)

            self.indi_metrics_matrices[metric] = np.asarray(indi_metric_matrix)

    def construct_group_metrics_matrix(self, metric_names):
        for metricidx, metric in enumerate(metric_names):
            group_metric_matrix = [rs.metrics[metric] for rs in self.runsets]
            self.group_metrics_matrices[metric] = group_metric_matrix


class datasetVeiwer(SetRunViewer):
    '''

    '''
    def __init__(self, runsets, dsetname):
        super().__init__(runsets, dsetname)

