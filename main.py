import os.path
import argparse
import torch, tqdm, clip, time
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report
import numpy as np
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.nn import CrossEntropyLoss
from utils import Mocheg_Dataset, Adapter

device = "cuda" if torch.cuda.is_available() else "cpu"

if __name__ == "__main__":
        parser = argparse.ArgumentParser(description="help")
        # parser.add_argument("--seed", type=int, help="this is seed number")
        parser.add_argument("--train", type=str, help="this is your train directory")
        parser.add_argument("--val", type=str, help="this is your val directory")
        parser.add_argument("--test", type=str, help="this is your test directory")
        # parser.add_argument("--shot", type=int, help='shots')
        parser.add_argument("--save_path", type=str, help="train model")


        args = parser.parse_args()

        print(device)

        print("Loading OpenAI CLIP.....")
        model, preprocess = clip.load('ViT-B/32', device, jit=False)
        # 分割数据集
        train_dataset = Mocheg_Dataset(model, preprocess, args.train)
        val_dataset = Mocheg_Dataset(model, preprocess, args.val)
        test_dataset = Mocheg_Dataset(model, preprocess, args.test)
        # 创建 DataLoader
        train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=128, shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False)


        # adapter = Adapter_Origin(num_classes=2).to(device)
        adapter = Adapter(num_classes=3).to(device)
        EPOCH = 20
        optimizer = AdamW(adapter.parameters(), lr=1e-3, eps=1e-4)
        # scheduler = CosineAnnealingLR(optimizer, EPOCH * len(train_loader))
        loss_func = CrossEntropyLoss()
        best_test_acc_in_epoch, patience_count, patience = 0, 0, 3
        num_trainable_params = sum(p.numel() for p in adapter.parameters() if p.requires_grad)
        print(f"Number of trainable parameters: {num_trainable_params}")
        for epoch in range(EPOCH):

            print("EPOCH: {} ".format(epoch + 1))
            for claim, evidence, img, label in tqdm.tqdm(train_loader):
                adapter.train()
                img_feat_0 = model.encode_image(img)
                claim_feat_0 = model.encode_text(claim)
                evidence_feat_0 = model.encode_text(evidence)

                img_feat = img_feat_0 / img_feat_0.norm(dim=-1, keepdim=True)
                claim_feat = claim_feat_0 / claim_feat_0.norm(dim=-1, keepdim=True)
                evidence_feat = evidence_feat_0 / evidence_feat_0.norm(dim=-1, keepdim=True)

                all_feat = torch.cat((img_feat, claim_feat, evidence_feat), dim=-1).to(device, torch.float32)

                logits = adapter(claim_feat_0.to(device, torch.float32),
                                       evidence_feat_0.to(device, torch.float32),
                                       img_feat_0.to(device, torch.float32), all_feat)

                loss = loss_func(logits, label)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                # scheduler.step()
                end_time = time.time()


            test_labels = []
            pred_labels = []

            print("Start to eval: ")
            with torch.no_grad():
                for claim, evidence, img, label in tqdm.tqdm(val_loader):
                    img_feat_1 = model.encode_image(img)
                    claim_feat_1 = model.encode_text(claim)
                    evidence_feat_1 = model.encode_text(evidence)

                    evidence_feat_0 = model.encode_text(evidence)

                    img_feat = img_feat_1 / img_feat_1.norm(dim=-1, keepdim=True)
                    claim_feat = claim_feat_1 / claim_feat_1.norm(dim=-1, keepdim=True)
                    evidence_feat = evidence_feat_1 / evidence_feat_1.norm(dim=-1, keepdim=True)

                    all_feat = torch.cat((img_feat, claim_feat, evidence_feat), dim=-1).to(device, torch.float32)
                    adapter.eval()
                    eval_logits = adapter(claim_feat_1.to(device, torch.float32),
                                     evidence_feat_1.to(device, torch.float32),
                                     img_feat_1.to(device, torch.float32), all_feat)

                    predictions = torch.argmax(torch.softmax(eval_logits, dim=1), dim=-1).cpu().numpy()
                    label = label.cpu().numpy()
                    accuracy = np.mean((label == predictions).astype(float)) * 100.

                    test_labels.extend(label)
                    pred_labels.extend(predictions)

            report = classification_report(test_labels, pred_labels, output_dict=True)
            print(report)
            print(f"Accuracy = {report['accuracy']:.3f}")
            print(f"F1 = {report['macro avg']['f1-score']:.3f}")
            epoch_acc = round(report['accuracy'], 3)
            if epoch_acc > best_test_acc_in_epoch:
                best_test_acc_in_epoch = epoch_acc
                print("Save best acc at EPOCH {}".format(epoch + 1))
                patience_count = 0
                if not os.path.exists(args.save_path):
                    os.makedirs(args.save_path)
                print("saving best model at Epoch {} | Accuracy {}".format(epoch+1, epoch_acc))
                torch.save(adapter.state_dict(),
                           args.save_path + "/adapter.pt")
            else:
                patience_count += 1

            if patience_count >= patience:
                print("Early stopping triggered")
                break  # 跳出训练循环

        print("best_acc found at: ", best_test_acc_in_epoch)
        with open(args.save_path + "/log.txt", 'w') as outf:
            outf.write(str(round(best_test_acc_in_epoch, 4)))