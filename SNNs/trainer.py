# import torch
# import numpy as np
import pandas as pd

# final_loss1=[]
# final_loss=[]
# epochs=[]
# def fit(train_loader, val_loader, model, loss_fn, optimizer, scheduler, n_epochs, cuda, log_interval, metrics=[],
#         start_epoch=0):
    

#     """
#     Loaders, model, loss function and metrics should work together for a given task,
#     i.e. The model should be able to process data output of loaders,
#     loss function should process target output of loaders and outputs from the model

#     Examples: Classification: batch loader, classification model, NLL loss, accuracy metric
#     Siamese network: Siamese loader, siamese model, contrastive loss
#     Online triplet learning: batch loader, embedding model, online triplet loss
#     """
#     for epoch in range(0, start_epoch):
#         scheduler.step()

#     for epoch in range(start_epoch, n_epochs):
#         scheduler.step()

#         # Train stage
#         train_loss, metrics = train_epoch(train_loader, model, loss_fn, optimizer, cuda, log_interval, metrics)
        
#         message = 'Epoch: {}/{}. Train set: Average loss: {:.4f}'.format(epoch + 1, n_epochs, train_loss)
#         for metric in metrics:
#             message += '\t{}: {}'.format(metric.name(), metric.value())

#         val_loss, metrics = test_epoch(val_loader, model, loss_fn, cuda, metrics)
#         val_loss /= len(val_loader)
        
#         message += '\nEpoch: {}/{}. Validation set: Average loss: {:.4f}'.format(epoch + 1, n_epochs,
#                                                                                  val_loss)
#         for metric in metrics:
#             message += '\t{}: {}'.format(metric.name(), metric.value())
        
        
#         epochs.append(epoch+1)
        
#         final_loss.append(val_loss)
#         final_loss1.append(train_loss)
# #         if epoch+1==99: 
# # #             d = {'epochs':epochs,'online_triple_2D':final_loss,'online_triple_train_2D':final_loss1}
# # #             df=pd.DataFrame(d)
# #             df = pd.read_csv("online_triple.csv")
# #             df['online_triple_16D'] =final_loss
# #             df['online_triple_train_16D'] = final_loss1        
# #             df.to_csv("online_triple.csv", encoding='utf-8', index=False)
#         print(message)



# def train_epoch(train_loader, model, loss_fn, optimizer, cuda, log_interval, metrics):
    
#     for metric in metrics:
#         metric.reset()

#     model.train()
#     losses = []
#     total_loss = 0

#     for batch_idx, (data, target) in enumerate(train_loader):
# #         print(data.type())
# #         target=target.long()
# #         print(target.type())
# #         print(target.shape)
# #         print(data.shape)

#         target = target if len(target) > 0 else None
#         if not type(data) in (tuple, list):
#             data = (data,)
#         if cuda:
#             data = tuple(d.cuda() for d in data)
#             if target is not None:
#                 target = target.cuda()


#         optimizer.zero_grad()
#         outputs = model(*data)

#         if type(outputs) not in (tuple, list):
#             outputs = (outputs,)

#         loss_inputs = outputs
#         if target is not None:
#             target = (target,)
#             loss_inputs += target

#         loss_outputs = loss_fn(*loss_inputs)
#         loss = loss_outputs[0] if type(loss_outputs) in (tuple, list) else loss_outputs
#         losses.append(loss.item())
#         total_loss += loss.item()
#         loss.backward()
#         optimizer.step()

#         for metric in metrics:
#             metric(outputs, target, loss_outputs)

#         if batch_idx % log_interval == 0:
#             message = 'Train: [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
#                 batch_idx * len(data[0]), len(train_loader.dataset),
#                 100. * batch_idx / len(train_loader), np.mean(losses))
#             for metric in metrics:
#                 message += '\t{}: {}'.format(metric.name(), metric.value())

#             print(message)
#             losses = []
   
#     total_loss /= (batch_idx + 1)
    
    


#     return total_loss, metrics


# def test_epoch(val_loader, model, loss_fn, cuda, metrics):
#     with torch.no_grad():
#         for metric in metrics:
#             metric.reset()
#         model.eval()
#         val_loss = 0
#         for batch_idx, (data, target) in enumerate(val_loader):
#             target = target if len(target) > 0 else None
#             if not type(data) in (tuple, list):
#                 data = (data,)
#             if cuda:
#                 data = tuple(d.cuda() for d in data)
#                 if target is not None:
#                     target = target.cuda()

#             outputs = model(*data)
# #             print(outputs,"pppppppppppppppppp")
#             if type(outputs) not in (tuple, list):
#                 outputs = (outputs,)
#             loss_inputs = outputs
#             if target is not None:
#                 target = (target,)
#                 loss_inputs += target

#             loss_outputs = loss_fn(*loss_inputs)
#             loss = loss_outputs[0] if type(loss_outputs) in (tuple, list) else loss_outputs
#             val_loss += loss.item()

#             for metric in metrics:
#                 metric(outputs, target, loss_outputs)

#     return val_loss, metrics
import torch
import numpy as np

final_loss1=[]
final_loss=[]
final_accuracy=[]
final_accuracy1=[]
epochs=[]
def fit(train_loader, val_loader, model, loss_fn, optimizer, scheduler, n_epochs, cuda, log_interval, metrics=[],
        start_epoch=0):
    """
    Loaders, model, loss function and metrics should work together for a given task,
    i.e. The model should be able to process data output of loaders,
    loss function should process target output of loaders and outputs from the model
    Examples: Classification: batch loader, classification model, NLL loss, accuracy metric
    Siamese network: Siamese loader, siamese model, contrastive loss
    Online triplet learning: batch loader, embedding model, online triplet loss
    """
    for epoch in range(0, start_epoch):
        scheduler.step()

    for epoch in range(start_epoch, n_epochs):
        scheduler.step()

        # Train stage
        train_loss, metrics = train_epoch(train_loader, model, loss_fn, optimizer, cuda, log_interval, metrics)
        final_loss1.append(train_loss)
        print(train_loss,"OOOOOOOOOOO")
        message = 'Epoch: {}/{}. Train set: Average loss: {:.4f}'.format(epoch + 1, n_epochs, train_loss)
        for metric in metrics:
            message += '\t{}: {}'.format(metric.name(), metric.value())

            final_accuracy1.append(metric.value())

        val_loss, metrics = test_epoch(val_loader, model, loss_fn, cuda, metrics)
        val_loss /= len(val_loader)

        message += '\nEpoch: {}/{}. Validation set: Average loss: {:.4f}'.format(epoch + 1, n_epochs,
                                                                                 val_loss)
        for metric in metrics:
            message += '\t{}: {}'.format(metric.name(), metric.value())
            final_accuracy.append(metric.value())
        epochs.append(epoch+1)

        final_loss.append(val_loss)
        print(message)
        if epoch+1==100:
#             d={'epochs':epochs,'pair_loss_4D':final_loss,'pair_train_loss_4D':final_loss1}
#             df=pd.DataFrame(d)
            df = pd.read_csv("softmax-net12.csv")
            df['online_triple_2D'] =final_loss
            df['online_triple_train_2D'] = final_loss1 
            df['online_triple_accuracy_2D'] =final_accuracy
            df['online_triple_train_accuracy_2D'] = final_accuracy1               
            df.to_csv("softmax-net12.csv", encoding='utf-8', index=False)
            



def train_epoch(train_loader, model, loss_fn, optimizer, cuda, log_interval, metrics):
    for metric in metrics:
        metric.reset()

    model.train()
    losses = []
    total_loss = 0

    for batch_idx, (data, target) in enumerate(train_loader):
        
        target = target if len(target) > 0 else None
       
        if not type(data) in (tuple, list):
            data = (data,)
        if cuda:
            data = tuple(d.cuda() for d in data)
            if target is not None:
                target = target.cuda()


        optimizer.zero_grad()
        outputs = model(*data)

        if type(outputs) not in (tuple, list):
            outputs = (outputs,)

        loss_inputs = outputs
        if target is not None:
            target = (target,)
            loss_inputs += target

        loss_outputs = loss_fn(*loss_inputs)
        loss = loss_outputs[0] if type(loss_outputs) in (tuple, list) else loss_outputs
        losses.append(loss.item())
        total_loss += loss.item()
        loss.backward()
        optimizer.step()

        for metric in metrics:

            metric(outputs, target, loss_outputs)

        if batch_idx % log_interval == 0:
            message = 'Train: [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                batch_idx * len(data[0]), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), np.mean(losses))
            for metric in metrics:
                message += '\t{}: {}'.format(metric.name(), metric.value())

            print(message)
            losses = []

    total_loss /= (batch_idx + 1)
    return total_loss, metrics


def test_epoch(val_loader, model, loss_fn, cuda, metrics):
    with torch.no_grad():
        for metric in metrics:
            metric.reset()
        model.eval()
        val_loss = 0
        for batch_idx, (data, target) in enumerate(val_loader):
            target = target if len(target) > 0 else None
            if not type(data) in (tuple, list):
                data = (data,)
            if cuda:
                data = tuple(d.cuda() for d in data)
                if target is not None:
                    target = target.cuda()

            outputs = model(*data)

            if type(outputs) not in (tuple, list):
                outputs = (outputs,)
            loss_inputs = outputs
            if target is not None:
                target = (target,)
                loss_inputs += target

            loss_outputs = loss_fn(*loss_inputs)
            loss = loss_outputs[0] if type(loss_outputs) in (tuple, list) else loss_outputs
            val_loss += loss.item()

            for metric in metrics:
                metric(outputs, target, loss_outputs)

    return val_loss, metrics