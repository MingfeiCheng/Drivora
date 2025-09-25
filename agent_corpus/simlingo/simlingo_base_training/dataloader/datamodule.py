import itertools
from typing import List

import hydra
import line_profiler
import numpy as np
import torch
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, LlavaNextProcessor

from simlingo_base_training.dataloader.carla_data import CARLA_Data
from simlingo_base_training.utils.custom_types import DrivingExample, DrivingInput, DrivingLabel
from simlingo_base_training.utils.projection import get_camera_intrinsics, get_camera_extrinsics


def encode_uint8(strings: List[str], common_length: int) -> torch.Tensor:
    max_len = max(len(s) for s in strings)
    assert max_len <= common_length, f"String is too long: {max_len} > {common_length}"
    padded_strings = [s.ljust(common_length, '\0') for s in strings]
    return torch.tensor([bytearray(s, 'utf-8') for s in padded_strings], dtype=torch.uint8)


class DataModule(LightningDataModule):
    def __init__(
        self,
        **cfg
    ):

        super().__init__()
        for key, value in cfg.items():
            setattr(self, key, value)

        self.cfg = cfg
       
        if 'resnet' in self.encoder_variant:
            self.processor = None
        
        elif self.encoder_variant is not None:
            self.processor = LlavaNextProcessor.from_pretrained(self.encoder_variant)

        if self.llm_variant is not None:
            if 'pythia' in self.llm_variant:
                self.llm_variant = f'EleutherAI/{self.llm_variant}'
                self.llm_tokenizer = AutoTokenizer.from_pretrained(self.llm_variant)

                # add eos token
                self.llm_tokenizer.add_eos_token = True
                self.llm_tokenizer.add_bos_token = True
                if self.llm_tokenizer.pad_token is None:
                    self.llm_tokenizer.add_special_tokens({'pad_token': '<|padding|>'})
            elif 'paligemma' in self.llm_variant:
                self.llm_variant = f'google/{self.llm_variant}'
                self.llm_tokenizer = AutoTokenizer.from_pretrained(self.llm_variant)
                self.llm_tokenizer.add_eos_token = True
                self.llm_tokenizer.add_bos_token = True
            elif 'TinyLlama' in self.llm_variant:
                self.llm_tokenizer = AutoTokenizer.from_pretrained(self.llm_variant, torch_dtype="auto",  trust_remote_code=True)
                self.llm_tokenizer.add_eos_token = True
                self.llm_tokenizer.add_bos_token = True
                if self.llm_tokenizer.pad_token is None:
                    self.llm_tokenizer.pad_token = "[PAD]"

            else:
                self.llm_tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf", token = "ADD_YOUR_TOKEN")
                self.llm_tokenizer.add_eos_token = True
                self.llm_tokenizer.add_bos_token = True
                if self.llm_tokenizer.pad_token is None:
                    self.llm_tokenizer.pad_token = self.llm_tokenizer.eos_token


    def setup(self, stage=None):
        if not self.predict:
            if self.train_partitions is not None:
                bucket_list = list(self.train_partitions.keys())
                bucket_proportions = [1.0] * len(bucket_list)
                sample_weights = list(self.train_partitions.values())
            elif self.predict:
                bucket_list = ['all']
                bucket_proportions = [1.0]
                sample_weights = [1.0]
            else:
                bucket_list = ['all']
                bucket_proportions = [1.0]
                sample_weights = [1.0]

            datasets = {}
            for bucket, bucket_proportion in zip(bucket_list, bucket_proportions):
                datasets[bucket] = CARLA_Data(
                    split="train",
                    bucket_name=bucket,
                    bucket_proportion=bucket_proportion,
                    **self.cfg,
                )

            self.train_dataset = torch.utils.data.ConcatDataset([datasets[bucket] for bucket in bucket_list])
            weights_train = [[sample_weights[i]] * datasets[bucket].__len__() for i, bucket in enumerate(bucket_list)]
            weights_train = list(itertools.chain.from_iterable(weights_train))
            num_samples_all = [datasets[bucket].__len__() // sample_weights[i] for i, bucket in enumerate(bucket_list)]
            num_samples = int(min(num_samples_all))
            print(f"Num samples: {num_samples}")
            print(f"Num samples all: {datasets['all'].__len__()}")
            # num_samples = int(datasets[bucket_list[-1]].__len__()//sample_weights[-1])
            self.sampler_train = torch.utils.data.WeightedRandomSampler(weights=weights_train, num_samples=num_samples, replacement=True)

            self.val_dataset = CARLA_Data(
                split="val",
                bucket_name="all",
                **self.cfg,
            )
            self.predict_dataset = None

        else:

            self.predict_dataset = CARLA_Data(
                split="train",
                bucket_name="all",
                **self.cfg,
            )


    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            # shuffle=self.shuffle,
            num_workers=self.num_workers,
            drop_last=True,
            collate_fn=self.dl_collate_fn,
            sampler=self.sampler_train,
            pin_memory=True,
        )

    def predict_dataloader(self):
        return DataLoader(
            self.predict_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            drop_last=True,
            collate_fn=self.dl_collate_fn,
            pin_memory=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            drop_last=True,
            collate_fn=self.dl_collate_fn,
            pin_memory=True,
        )


    @line_profiler.profile
    def dl_collate_fn(self, data):
        # print(f"collate_fn")
        B = len(data)
        T = data[0]['rgb'].shape[0]
        N = data[0]['rgb'].shape[1]
        C = data[0]['rgb'].shape[2]
        H = data[0]['rgb'].shape[3]
        W = data[0]['rgb'].shape[4]
        F = 11 # future WPs TODO: add to config
        dT = 0.2 # future time steps TODO: add to config

        image_sizes = None

        if self.encoder == 'llavanext':
            images_batch_list = torch.tensor(np.asarray([data[i]["rgb"] for i in range(len(data))]))
            # move T and N to batch dimension
            images_batch_list = images_batch_list.view(B*T*N, C, H, W)
            images_batch_list = list(images_batch_list)
            images_processed = self.processor.image_processor(images_batch_list, return_tensors="pt", image_grid_pinpoints=[[336,672]])
            images_pixel = images_processed['pixel_values']
            image_sizes = images_processed['image_sizes']


            if not self.use_global_img:
                # remove global patch
                images_pixel = images_pixel[:,1:]

            num_patches = images_pixel.shape[1]
            new_height = images_pixel.shape[3]
            new_width = images_pixel.shape[4]
            images_pixel = images_pixel.view(B, T, N, num_patches, C, new_height, new_width)
        else:
            images_pixel = torch.tensor(np.asarray([data[i]["rgb"] for i in range(len(data))])).half()

        return DrivingExample(
            driving_input=DrivingInput(
                camera_images=images_pixel,  # [B, T, N, C, H, W] uint8 [0, 255]
                image_sizes=image_sizes,
                camera_intrinsics = torch.repeat_interleave(get_camera_intrinsics(W, H, 110).unsqueeze(0), B * N, dim=0).view(B, N, 3, 3).float(),
                camera_extrinsics = torch.repeat_interleave(get_camera_extrinsics().unsqueeze(0), B * N, dim=0).view(B, N, 4, 4).float(),
                vehicle_speed=torch.tensor(np.asarray([[data[i]["speed"]] for i in range(len(data))])).float(),  # [B, S] float32
                map_route=torch.tensor(np.asarray([data[i]["map_route"] for i in range(len(data))])).float(),  # [B, 3, RH, RW] uint8 [0, 255]
                target_point=torch.tensor(np.asarray([data[i]["target_point"] for i in range(len(data))])).float(),  # [B, 2] float32
            ),
            driving_label=DrivingLabel(
                time_delta_sec=torch.tensor([dT*i for i in range(F)]).repeat(B, 1).float(), # [B, F] 0-2 sec 0.2s apart
                waypoints=torch.tensor(np.asarray([data[i]["waypoints"] for i in range(len(data))])).float(), # [B, F, 2] 11 future waypoints 0.2s apart
                waypoints_1d=torch.tensor(np.asarray([data[i]["waypoints_1d"] for i in range(len(data))])).float(), # [B, F, 2] 11 future waypoints 0.2s apart
                route_adjusted=torch.tensor(np.asarray([data[i]["route_adjusted"] for i in range(len(data))])).float(), # [B, 3, RH, RW] uint8 [0, 255]
            ),
            run_id=encode_uint8([data[i]["measurement_path"] for i in range(len(data))], 1000),  # [B] str
            timestamp=torch.zeros(B, dtype=torch.int64),  # [B] float32
        )

    def dl_collate_fn_val(self, data):
        pass

    def dl_collate_fn_test(self, data):
        pass