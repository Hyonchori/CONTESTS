import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

DIR_2020 = "235731_북극 해빙예측 AI 경진대회_data/data/"
DIR_2021 = "235731_북극_해빙예측_AI_경진대회_data_v2/data_v2/"

def get_img_batch(img_path):
    imgs = []
    mask = None
    for path in img_path:
        img = np.load(path)[:, :, 0] / 250.
        imgs.append(img)

        if mask is None:
            mask = np.load(path)[:, :, 3]
            mask = np.where(mask==1, 0, 1)

    imgs = np.array(imgs)
    imgs = torch.from_numpy(imgs)
    shape = imgs.size()
    imgs = imgs.view(1, 1, shape[0], shape[1], shape[2])
    return imgs, mask

def vis_seq(imgs):
    for i in range(imgs.shape[0]):
        plt.imshow(imgs[i])
        plt.axis("off")
        plt.show()

def vis_seq_batch(imgs):
    imgs = imgs[0]
    plt.figure(figsize=(16, 3))
    for i in range(imgs.shape[0]):
        plt.subplot(1, imgs.shape[0], i+1)
        plt.imshow(imgs[i])
        plt.axis("off")
    plt.show()

def get_public_pred(model, sub_name):
    data_pd = pd.read_csv(DIR_2020 + "public_weekly_test.csv")[-12: ]
    data_path = DIR_2020 + "weekly_train/" + data_pd["week_file_nm"].values
    model.eval()

    imgs, mask = get_img_batch(data_path)
    pred = model(imgs.float())[0, 0].detach().numpy() * 250.
    pred = pred.mask
    vis_seq(pred)

    result = pred.reshape([12, -1])

    submission = pd.read_csv(DIR_2020 + "sample_submission.csv")
    sub_2020 = submission.loc[:11, ["week_start"]].copy()
    sub_2021 = submission.loc[12: ].copy()

    sub_2020 = pd.concat([sub_2020, (pd.DataFrame(result))], axis=1)
    sub_2021.columns = sub_2020.columns
    submission = pd.concat([sub_2020, sub_2021])
    submission.to_csv(sub_name, index=False)

def get_private_pred(model, sub_name):
    public_pd = pd.read_csv(DIR_2020 + "public_weekly_test.csv")[-12: ]
    public_path = DIR_2020 + "weekly_train/" + public_pd["week_file_nm"].values

    private_pd = pd.read_csv(DIR_2021 + "private_weekly_test.csv")[-12: ]
    private_path = DIR_2021 + "weekly_train/" + private_pd["week_file_nm"].values

    public_imgs, mask = get_img_batch(public_path)
    public_pred = model(public_imgs.float())[0, 0].detach().numpy() * 250.
    public_pred *= mask
    public_result = public_pred.reshape([12, -1])
    print("\n---\nPublic Prediction: ")
    vis_seq_batch(public_imgs.detach().numpy()[:, 0])
    vis_seq_batch(public_pred[np.newaxis, ...])
    vis_seq(public_pred)

    private_imgs, mask = get_img_batch(private_path)
    private_pred = model(private_imgs.float())[0, 0].detach().numpy() * 250.
    private_pred *= mask
    private_result = private_pred.reshape([12, -1])
    print("\n---\nPrivate Prediction: ")
    vis_seq_batch(private_imgs.detach().numpy()[:, 0])
    vis_seq_batch(private_pred[np.newaxis, ...])
    vis_seq(private_pred)

    submission = pd.read_csv(DIR_2021 + "sample_submission.csv")
    sub_2020 = submission.loc[:11, ["week_start"]].copy()
    sub_2021 = submission.loc[12:, ["week_start"]].copy()
    sub_2020 = pd.concat([sub_2020, (pd.DataFrame(public_result))], axis=1)
    sub_2021 = pd.concat([sub_2021, (pd.DataFrame(private_result, index=list(range(12, 24))))], axis=1)
    submission_ = pd.concat([sub_2020, sub_2021])
    submission_.to_csv(sub_name, index=False)