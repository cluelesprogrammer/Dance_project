import models.models as models
import torch

def get_model(model_type, train_features, trained_model=True, paf_stages=4, conf_stages=2, batch_norm=True, final_activations=(None, None)):
        if (model_type == 'old'):
                #loading the older model
                model = models.bodypose_model(trained_model=trained_model).float()
        else:
                model = models.new_bodypose_model(paf_stages, conf_stages, train_features, batch_norm, final_activations).float()
        #model.apply(init_weights)
        return model

def init_weights(m):
        if (type(m) in [torch.nn.Conv2d, torch.nn.Linear]):
                torch.nn.init.xavier_uniform(m.weight)
                m.bias.data.fill_(0.01)


