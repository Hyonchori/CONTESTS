import torch
import torch.optim as optim
import matplotlib.pyplot as plt

from data import get_dataloader
from model import EncoderAndDecoder
from train import train_model
from infer import get_private_pred

train_dataloader, valid_dataloader = get_dataloader()
model = EncoderAndDecoder()

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model.to(device)

optimizer = optim.Adam(model.parameters(), lr=0.0004)
exp_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.95)

save_dir = "melt_pred.pth"
start_epoch, end_epoch = 0, 200
train_losses, valid_losses = train_model(model, optimizer, exp_scheduler, device, save_dir,
                                         start_epoch, end_epoch, train_dataloader, valid_dataloader)

plt.plot(train_losses["mae"][10: ])
plt.plot(valid_losses["mae"][10: ])
plt.show()

plt.plot(train_losses["f1"][10: ])
plt.plot(valid_losses["f1"][10: ])
plt.show()

plt.plot(train_losses["mae_over_f1"][10: ])
plt.plot(valid_losses["mae_over_f1"][10: ])
plt.show()

predictor = EncoderAndDecoder().load_state_dict(torch.load(save_dir))
predictor.eval()

get_private_pred(model, "submission.csv")