import os
import csv
import copy
import random
import matplotlib.pyplot as plt
import pandas as pd

from generateChromosome import Chromosome

Dataset_save_path = os.path.dirname(os.getcwd())+'/result'

class GeneticAlgorithm:

    def __init__(self, chr_base, chr_len, max_mut_num, pm, pc, pop_size, thres):
        """
        Algorithm Initializaiton
        :param chr_base: chromosome_base
        :param chr_len: chromosome_length
        :param max_mut_num: max_mutation_number
        :param pm: probability_mutation
        :param pc: probability_crossover
        :param pop_size: population_size
        :param thres: stop_threshold
        :return:
        """
        self.chr_base = chr_base
        self.chr_len = chr_len
        self.max_mut_num = max_mut_num
        self.pm = pm
        self.pc = pc
        self.pop_size = pop_size
        self.thres = thres

        self.pop = []
        self.pop_saved = []
        self.higher_rate = 0
        self.bests = [0]*100000
        self.g_best = 0

    def ga(self):
        """
        main
        :return:
        """
        self.init_pop()
        best = self.find_best()
        self.g_best = copy.deepcopy(best)
        i = 0
        while self.higher_rate < self.thres:
            self.cross()
            self.mutation()
            self.select()
            count = 0
            for j in range(self.pop_size):
                tmp = {}
                if self.pop[j].label[0] > 0:
                    tmp['Chr'] = self.pop[j].Chr[0]
                    tmp['TA_content'] = self.pop[j].TA_content
                    tmp['label'] = self.pop[j].label
                    self.pop_saved.append(tmp)
                    count += 1
            self.higher_rate = count/self.pop_size
            best = self.find_best()
            self.bests[i] = best
            if best.label >= self.g_best.label and best.y > self.g_best.y:
                self.g_best = copy.deepcopy(best)
            i += 1
            print("The {} generation: higer_rate = {}".format(i, self.higher_rate))
        # # plt
        # plt.figure(1)
        # x = range(len(self.higher_rate))
        # plt.plot(x, self.higher_rate)
        # plt.ylabel('generations')
        # plt.xlabel('higher_individual_rate')
        # plt.show()

    def init_pop(self):
        """
        Population Initialization
        :return:
        """
        for _ in range(self.pop_size):
            chromosome = Chromosome(self.chr_base, self.chr_len)
            self.pop.append(chromosome)

    def cross(self):
        """
        Crossover
        :return:
        """
        for i in range(int(self.pop_size / 2)):
            if self.pc > random.random():
                # randon select 2 chromosomes in pops
                i = 0
                j = 0
                while i == j:
                    i = random.randint(0, self.pop_size-1)
                    j = random.randint(0, self.pop_size-1)
                pop_i = self.pop[i]
                pop_j = self.pop[j]

                # select cross index
                pos = random.randint(0, pop_i.chr_len - 1)

                # get new code
                new_pop_i = []
                new_pop_j = []
                tmp_pop_i = pop_i.Chr[0][0: pos] + pop_j.Chr[0][pos: pop_i.chr_len]
                tmp_pop_j = pop_j.Chr[0][0: pos] + pop_i.Chr[0][pos: pop_i.chr_len]
                new_pop_i.append(tmp_pop_i)
                new_pop_j.append(tmp_pop_j)

                pop_i.Chr = new_pop_i
                pop_j.Chr = new_pop_j

    def mutation(self):
        """
        Mutation
        :return:
        """
        for i in range(self.pop_size):
            if self.pm > random.random():
                pop = self.pop[i]
                tmp = list(pop.Chr[0])
                # select mutation index
                mut_num = random.randint(1, self.max_mut_num)
                index = random.sample(range(0, self.chr_len), mut_num)
                for i in range(len(index)):
                    mut_base_selection = [s for s in 'ATCG' if s != tmp[index[i]]]
                    mut_base_type = random.choice(mut_base_selection)
                    tmp[index[i]] = mut_base_type
                new_pop = []
                new_pop.append("".join(tmp))
                pop.Chr = new_pop

    def select(self):
        """
        Roulette selection
        :return:
        """
        sum_f = 0
        for i in range(self.pop_size):
            self.pop[i].func()

        max = self.pop[0].y
        for i in range(self.pop_size):
            if self.pop[i].label[0] > 0 and self.pop[i].y > max:
                max = self.pop[i].y

        # roulette
        for i in range(self.pop_size):
            sum_f += self.pop[i].y
        p = [0] * self.pop_size
        for i in range(self.pop_size):
            p[i] = self.pop[i].y / sum_f
        q = [0] * self.pop_size
        q[0] = 0
        for i in range(self.pop_size):
            s = 0
            for j in range(0, i+1):
                s += p[j]
            q[i] = s
        # start roulette
        v = []
        for i in range(self.pop_size):
            r = random.random()
            if r < q[0]:
                v.append(self.pop[0])
            for j in range(1, self.pop_size):
                if q[j - 1] < r <= q[j]:
                    v.append(self.pop[j])
        self.pop = v
        

    def find_best(self):
        """
        find the best individual of population
        :return:
        """
        best = copy.deepcopy(self.pop[0])
        for i in range(self.pop_size):
            if self.pop[i].label >= best.label and self.pop[i].y > best.y:
                best = copy.deepcopy(self.pop[i])
        return best

if __name__ == '__main__':
    
    chr_base = 'ATCAAAATTTAACTGTTCTAACCCCTACTTGACAGCAATATATAAACAGAAGGAAGCTGCCCTGTCTTAAACCTTTTTTTTTATCATCAT'
    chr_len = 90
    max_mut_num = 10
    prob_mutation = 0.05
    prob_crossover = 0.8
    pop_size = 200
    thres = 0.95
    
    algorithm = GeneticAlgorithm(chr_base, chr_len, max_mut_num, prob_mutation, prob_crossover, pop_size, thres)
    algorithm.ga()

    Seq_Optimized_GA_fn = Dataset_save_path+"/Optimized_sequence.csv"
    # for i in range(len(algorithm.pop_saved)):
    #     with open(Seq_Optimized_GA_fn, mode='a') as outfile:
    #             outfile = csv.writer(outfile, delimiter=',')
    #             outfile.writerow([algorithm.pop_saved[i].Chr[0], algorithm.pop_saved[i].TA_content, algorithm.pop_saved[i].label])

    saved = pd.DataFrame(algorithm.pop_saved)
    saved.to_csv(Seq_Optimized_GA_fn)