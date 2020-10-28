import argparse
import os
import pathlib
import numpy as np
import torch
import torch.nn as nn
import torch.multiprocessing as _mp
from torchvision.datasets import MNIST
import torchvision.transforms as transforms
from model import Net
from trainer import Trainer
from utils import get_optimizer
import time

mp = _mp.get_context('spawn')


class Worker(mp.Process):
    def __init__(self, batch_size, epoch, max_epoch, train_data, test_data, population, finish_tasks,
                 device):
        super().__init__()
        self.epoch = epoch
        self.population = population
        self.finish_tasks = finish_tasks
        self.max_epoch = max_epoch
        self.batch_size = batch_size
        self.device = device
        model = Net().to(device)
        optimizer = get_optimizer(model)
        self.trainer = Trainer(model=model,
                               optimizer=optimizer,
                               loss_fn=nn.CrossEntropyLoss(),
                               train_data=train_data,
                               test_data=test_data,
                               batch_size=self.batch_size,
                               device=self.device)

    def run(self):
        while True:
            if self.epoch.value > self.max_epoch:
                break
            # Train
            task = self.population.get()
            self.trainer.set_id(task['id'])
            checkpoint_path = "checkpoints/task-%03d.pth" % task['id']
            if os.path.isfile(checkpoint_path):
                self.trainer.load_checkpoint(checkpoint_path)
            try:
                if self.epoch.value > self.max_epoch: # In case you get here before epoch is updated
                  break
                #print(self.device)
                self.trainer.train()
                score = 0
                self.trainer.save_checkpoint(checkpoint_path)
                self.finish_tasks.put(dict(id=task['id'], score=score))
            except KeyboardInterrupt:
                break

class Sentinel(mp.Process):
    def __init__(self, epoch, max_epoch, population, finish_tasks):
        super().__init__()
        self.epoch = epoch
        self.population = population
        self.finish_tasks = finish_tasks
        self.max_epoch = max_epoch

    def run(self):
        while True:
            if self.epoch.value > self.max_epoch:
                break
            if self.population.empty() and self.finish_tasks.full():
                print("End Training")
                tasks = []
                while not self.finish_tasks.empty():
                    tasks.append(self.finish_tasks.get())
                with self.epoch.get_lock():
                    self.epoch.value += 1
                for task in tasks:
                    self.population.put(task)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Population Based Training")
    parser.add_argument("--device", type=str, default='cuda',
                        help="")
    parser.add_argument("--population_size", type=int, default=48,
                        help="")
    parser.add_argument("--num_workers", type=int, default=6,
                        help="")                        
    parser.add_argument("--batch_size", type=int, default=200,
                        help="")

    # Hyperparameters
    args = parser.parse_args()
    num_workers = args.num_workers
    population_size = args.population_size
    batch_size = args.batch_size
    max_epoch = 2
    n_GPU = 1

    # Multiprocessing settings
    # mp.set_start_method("spawn")
    mp = mp.get_context('forkserver')
    device = ['cuda:0', 'cuda:1', 'cuda:2', 'cuda:3']        
    device = device[:n_GPU]
    print(f"Using Device: {device}")

    # Data
    train_data_path = test_data_path = './data'
    train_data = MNIST(train_data_path, True, transforms.ToTensor(), download=True)
    test_data = MNIST(test_data_path, False, transforms.ToTensor(), download=True)

    # Create queues for work to be done and work done
    epoch = mp.Value('i', 0) # shared value across workers
    population = mp.Queue(maxsize=population_size)
    finish_tasks = mp.Queue(maxsize=population_size)

    # Fill up queue
    for i in range(population_size):
        population.put(dict(id=i, score=0))

    # Create workers and 'Sentinel' to count epochs and reload queues at end of each 
    workers = [Worker(batch_size, epoch, max_epoch, train_data, test_data, population, finish_tasks, device[i%len(device)])
               for i in range(num_workers)]
    workers.append(Sentinel(epoch, max_epoch, population, finish_tasks))

    t_start = time.time()
    [w.start() for w in workers] # Start workers
    [w.join() for w in workers]  # Gather up threads


    print(f'Training with {str(num_workers)} workers:\t{time.time() - t_start}\n\n')

    # task = []
    # while not finish_tasks.empty():
    #     task.append(finish_tasks.get())
    # while not population.empty():
    #     task.append(population.get())
    # task = sorted(task, key=lambda x: x['score'], reverse=True)
    # print('best score on', task[0]['id'], 'is', task[0]['score'])

    # Checkpoint saving for picking up models
    # hyper_params = {'optimizer': ["lr", "momentum"], "batch_size": True}
    # pathlib.Path('checkpoints').mkdir(exist_ok=True)
    # checkpoint_str = "checkpoints/task-%03d.pth"