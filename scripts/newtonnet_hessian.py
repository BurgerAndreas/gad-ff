import os
import argparse

import torch
from torch.optim import Adam
import yaml
import sys
import hydra
from omegaconf import DictConfig, OmegaConf

import newtonnet1 as newtonnet
from newtonnet1.layers.activations import get_activation_by_string
from newtonnet1.models import NewtonNet

from newtonnet1.train import Trainer
from newtonnet1.data import parse_train_test
from newtonnet1.data import parse_ani_data
from newtonnet1.data import parse_methane_data
from newtonnet1.data import parse_t1x_data

# torch.autograd.set_detect_anomaly(True)
torch.set_default_tensor_type(torch.DoubleTensor)

# argument parser description
parser = argparse.ArgumentParser(
    description="This is a pacakge to train NewtonNet on a given data."
)
parser.add_argument(
    "-c",
    "--config",
    type=str,
    required=False,
    help="The path to the Yaml configuration file.",
    default="create_gad_dataset_composition.yaml",
)
parser.add_argument(
    "-p",
    "--parser",
    type=str,
    required=False,
    default="t1x",
    help="The name of dataset to select the appropriate parser. We provide data parsers for 'md17', 'ccsd', 'ani', 'hydroggen' and 'methane' data sets."
    "For all other data sets do not specify.",
)

# define arguments
args = parser.parse_args()
config = args.config
parser = args.parser

@hydra.main(version_base=None, config_path="../configs", config_name=config)
def main(settings: DictConfig):
    
    # Convert DictConfig to dict and resolve environment variables
    settings_dict = OmegaConf.to_container(settings, resolve=True)
    print("-"*40)
    print("settings_dict: ", yaml.dump(settings_dict))
    
    # device
    if isinstance(settings_dict["general"]["device"], list):
        device = [torch.device(item) for item in settings_dict["general"]["device"]]
    else:
        device = [torch.device(settings_dict["general"]["device"])]

    # data
    print(f"Loading data from {parser} dataset")
    if parser in ["md17", "ccsd"]:
        train_gen, val_gen, test_gen, tr_steps, val_steps, test_steps, normalizer = (
            parse_train_test(settings_dict, device[0])
        )
        train_mode = "energy/force"
        print("data set:", "one of md17 data sets or a generic one.")
    elif parser in ["ani"]:
        (
            train_gen,
            val_gen,
            test_gen,
            tr_steps,
            val_steps,
            test_steps,
            n_train_data,
            n_val_data,
            n_test_data,
            normalizer,
            test_energy_hash,
        ) = parse_ani_data(settings_dict, device[0])
        train_mode = "energy"
        print("data set:", "ANI")
    elif parser in ["methane"]:
        (
            train_gen,
            val_gen,
            test_gen,
            tr_steps,
            val_steps,
            test_steps,
            n_train_data,
            n_val_data,
            n_test_data,
            normalizer,
            test_energy_hash,
        ) = parse_methane_data(settings_dict, device[0])
        train_mode = "energy/force"
        print("data set:", "Methane Combustion")
    elif parser in ["hydrogen"]:
        (
            train_gen,
            val_gen,
            test_gen,
            tr_steps,
            val_steps,
            test_steps,
            n_train_data,
            n_val_data,
            n_test_data,
            normalizer,
            test_energy_hash,
        ) = parse_methane_data(settings_dict, device[0])
        train_mode = "energy/force"
        print("data set:", "Methane Combustion")
    elif parser in ["t1x"]:
        train_gen, val_gen, test_gen, tr_steps, val_steps, test_steps, normalizer = (
            parse_t1x_data(settings_dict, device[0])
        )
        train_mode = "energy/force"
        print("data set:", "Transition1x")
    else:
        raise ValueError(f"Invalid parser: {parser}")

    print("normalizer: ", normalizer)

    # model
    # activation function
    activation = get_activation_by_string(settings_dict["model"]["activation"])

    model = NewtonNet(
        resolution=settings_dict["model"]["resolution"],
        n_features=settings_dict["model"]["n_features"],
        activation=activation,
        n_interactions=settings_dict["model"]["n_interactions"],
        dropout=settings_dict["training"]["dropout"],
        max_z=settings_dict["model"]["max_z"],
        cutoff=settings_dict["data"]["cutoff"],  ## data cutoff
        cutoff_network=settings_dict["model"]["cutoff_network"],
        normalizer=normalizer,
        normalize_atomic=settings_dict["model"]["normalize_atomic"],
        requires_dr=settings_dict["model"]["requires_dr"],
        device=device[0],
        create_graph=True,
        shared_interactions=settings_dict["model"]["shared_interactions"],
        return_hessian=settings_dict["model"]["return_hessian"],
        double_update_latent=settings_dict["model"]["double_update_latent"],
        layer_norm=settings_dict["model"]["layer_norm"],
    )

    # laod pre-trained model
    if settings_dict["model"]["pre_trained"]:
        model_path = settings_dict["model"]["pre_trained"]
        model.load_state_dict(torch.load(model_path)["model_state_dict"])

    # optimizer
    trainable_params = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = Adam(
        trainable_params,
        lr=settings_dict["training"]["lr"],
        weight_decay=settings_dict["training"]["weight_decay"],
    )

    # loss
    w_energy = settings_dict["model"]["w_energy"]
    w_force = settings_dict["model"]["w_force"]
    w_f_mag = settings_dict["model"]["w_f_mag"]
    w_f_dir = settings_dict["model"]["w_f_dir"]
    lambda_l1 = settings_dict["model"]["lambda_l1"]

    def custom_loss(
        preds,
        batch_data,
        params,
        w_e=w_energy,
        w_f=w_force,
        w_fm=w_f_mag,
        w_fd=w_f_dir,
        lambda_l1=lambda_l1,
    ):
        # compute the mean squared error on the energies
        diff_energy = preds["E"] - batch_data["E"]
        assert diff_energy.shape[1] == 1
        err_sq_energy = torch.mean(diff_energy**2)
        err_sq = w_e * err_sq_energy

        # compute the mean squared error on the forces
        diff_forces = preds["F"] - batch_data["F"]
        err_sq_forces = torch.mean(diff_forces**2)
        err_sq = err_sq + w_f * err_sq_forces

        # compute the mean square error on the force magnitudes
        if w_fm > 0:
            diff_forces = torch.norm(preds["F"], p=2, dim=-1) - torch.norm(
                batch_data["F"], p=2, dim=-1
            )
            err_sq_mag_forces = torch.mean(diff_forces**2)
            err_sq = err_sq + w_fm * err_sq_mag_forces

        if w_fd > 0:
            cos = torch.nn.CosineSimilarity(dim=-1, eps=1e-6)
            direction_diff = 1 - cos(preds["F_latent"], batch_data["F"])
            # direction_diff = direction_diff * torch.norm(batch_data["F"], p=2, dim=-1)
            direction_loss = torch.mean(direction_diff)
            err_sq = err_sq + w_fd * direction_loss

        if lambda_l1 > 0:
            for param in params:
                err_sq = err_sq + lambda_l1 * param.norm(1)

        if settings_dict["checkpoint"]["verbose"]:
            print(
                "\n", " " * 8, "energy loss: ", err_sq_energy.detach().cpu().numpy(), "\n"
            )
            print(" " * 8, "force loss: ", err_sq_forces.detach().cpu().numpy(), "\n")

            if w_fm > 0:
                print(" " * 8, "force mag loss: ", err_sq_mag_forces, "\n")

            if w_fd > 0:
                print(" " * 8, "direction loss: ", direction_loss.detach().cpu().numpy())

        return err_sq

    # training
    trainer = Trainer(
        model=model,
        loss_fn=custom_loss,
        optimizer=optimizer,
        requires_dr=settings_dict["model"]["requires_dr"],
        device=device,
        yml_path=settings_dict["general"]["me"],
        output_path=settings_dict["general"]["output"],
        script_name=settings_dict["general"]["driver"],
        lr_scheduler=settings_dict["training"]["lr_scheduler"],
        energy_loss_w=w_energy,
        force_loss_w=w_force,
        loss_wf_decay=settings_dict["model"]["wf_decay"],
        lambda_l1=lambda_l1,
        checkpoint_log=settings_dict["checkpoint"]["log"],
        checkpoint_val=settings_dict["checkpoint"]["val"],
        checkpoint_test=settings_dict["checkpoint"]["test"],
        checkpoint_model=settings_dict["checkpoint"]["model"],
        verbose=settings_dict["checkpoint"]["verbose"],
        hooks=settings_dict["hooks"],
        mode=train_mode,
    )

    trainer.train(
        train_generator=train_gen,
        epochs=settings_dict["training"]["epochs"],
        steps=tr_steps,
        val_generator=val_gen,
        val_steps=val_steps,
        irc_generator=None,
        irc_steps=None,
        test_generator=test_gen,
        test_steps=test_steps,
        clip_grad=1.0,
    )

    print("done!")

if __name__ == "__main__":
    main()
