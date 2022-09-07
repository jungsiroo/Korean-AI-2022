import time
import torch

def trainer(mode, config, dataloader, optimizer, model, criterion, metric, train_begin_time, device):

    log_format = "[INFO] step: {:4d}/{:4d}, loss: {:.6f}, " \
                              "cer: {:.2f}, elapsed: {:.2f}s {:.2f}m {:.2f}h, lr: {:.6f}"
    total_num = 0
    epoch_loss_total = 0.
    print(f'[INFO] {mode} Start')
    epoch_begin_time = time.time()
    cnt = 0
    for inputs, targets, input_lengths, target_lengths in dataloader:
        begin_time = time.time()

        optimizer.zero_grad()
        inputs = inputs.to(device)
        targets = targets.to(device)
        input_lengths = input_lengths.to(device)
        target_lengths = torch.as_tensor(target_lengths).to(device)
        model = model.to(device)
        teacher_forcing_ratio = 1.0 if mode =='train' else 0.0
        encoder_log_probs, encoder_outputs, encoder_output_lengths = model.encoder(inputs, input_lengths)
        if mode == 'train':
          outputs = model.decoder(targets=targets, encoder_outputs=encoder_outputs, teacher_forcing_ratio=teacher_forcing_ratio)
        else:
          outputs = model.decoder(encoder_outputs=encoder_outputs, teacher_forcing_ratio=teacher_forcing_ratio)
        max_target_length = max(target_lengths)  # minus the start of sequence symbol
        outputs = outputs[:, :max_target_length, :].to(device)
        loss= criterion(
            encoder_log_probs=encoder_log_probs.transpose(0, 1),
            decoder_log_probs=outputs.contiguous().view(-1, outputs.size(-1)),
            output_lengths=encoder_output_lengths,
            target_lengths=target_lengths,
            targets=targets[:, 1:],
        )

        y_hats = outputs.max(-1)[1]

        if mode == 'train':
            optimizer.zero_grad()
            loss.backward()
            optimizer.step(model)

        total_num += int(input_lengths.sum())
        epoch_loss_total += loss.item()
        
        torch.cuda.empty_cache()

        if cnt % config.print_every == 0:

            current_time = time.time()
            elapsed = current_time - begin_time
            epoch_elapsed = (current_time - epoch_begin_time) / 60.0
            train_elapsed = (current_time - train_begin_time) / 3600.0
            cer = metric(targets[:, 1:], y_hats)
            print(log_format.format(
                cnt, len(dataloader), loss,
                cer, elapsed, epoch_elapsed, train_elapsed,
                optimizer.get_lr(),
            ))
        cnt += 1
    return model, epoch_loss_total/len(dataloader), metric(targets[:, 1:], y_hats)