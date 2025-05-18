# Description of Neural Network Models for Franka Cabinet
Note that in this terminology, "network" is the neural network, and "model" is contains the network, settings and more stuff

## Constructing outputs frm NN

Network via A2CBuilder

```python

class ModelA2CContinuousLogStd(BaseModel):
    def __init__(self, network):
        BaseModel.__init__(self, 'a2c')
        self.network_builder = network

    class Network(BaseModelNetwork):
        def __init__(self, a2c_network, **kwargs):
            BaseModelNetwork.__init__(self, **kwargs)
            self.a2c_network = a2c_network

        def is_rnn(self):
            return self.a2c_network.is_rnn()

        def get_value_layer(self):
            return self.a2c_network.get_value_layer()

        def get_default_rnn_state(self):
            return self.a2c_network.get_default_rnn_state()

        def forward(self, input_dict):
            is_train = input_dict.get('is_train', True)
            prev_actions = input_dict.get('prev_actions', None)
            input_dict['obs'] = self.norm_obs(input_dict['obs'])
            mu, logstd, value, states = self.a2c_network(input_dict)
            sigma = torch.exp(logstd)
            distr = torch.distributions.Normal(mu, sigma, validate_args=False)
            if is_train:
                entropy = distr.entropy().sum(dim=-1)
                prev_neglogp = self.neglogp(prev_actions, mu, sigma, logstd)
                result = {
                    'prev_neglogp' : torch.squeeze(prev_neglogp),
                    'values' : value,
                    'entropy' : entropy,
                    'rnn_states' : states,
                    'mus' : mu,
                    'sigmas' : sigma
                }                
                return result
            else:
                selected_action = distr.sample()
                neglogp = self.neglogp(selected_action, mu, sigma, logstd)
                result = {
                    'neglogpacs' : torch.squeeze(neglogp),
                    'values' : self.denorm_value(value),
                    'actions' : selected_action,
                    'rnn_states' : states,
                    'mus' : mu,
                    'sigmas' : sigma
                }
                return result

        def neglogp(self, x, mean, std, logstd):
            return 0.5 * (((x - mean) / std)**2).sum(dim=-1) \
                + 0.5 * np.log(2.0 * np.pi) * x.size()[-1] \
                + logstd.sum(dim=-1)
```

## Model

when printing `nn_model`

```
Model Structure:
Network(
  (value_mean_std): RunningMeanStd()
  (running_mean_std): RunningMeanStd()
  (a2c_network): Network(
    (actor_cnn): Sequential()
    (critic_cnn): Sequential()
    (actor_mlp): Sequential(
      (0): Linear(in_features=23, out_features=256, bias=True)
      (1): ELU(alpha=1.0)
      (2): Linear(in_features=256, out_features=128, bias=True)
      (3): ELU(alpha=1.0)
      (4): Linear(in_features=128, out_features=64, bias=True)
      (5): ELU(alpha=1.0)
    )
    (critic_mlp): Sequential()
    (value): Linear(in_features=64, out_features=1, bias=True)
    (value_act): Identity()
    (mu): Linear(in_features=64, out_features=9, bias=True)
    (mu_act): Identity()
    (sigma_act): Identity()
  )
)

```

```
<class 'rl_games.algos_torch.models.ModelA2CContinuousLogStd.Network'>

Model attributes:
  training: False
  _parameters: OrderedDict()
  _buffers: OrderedDict()
  _non_persistent_buffers_set: set()
  _backward_pre_hooks: OrderedDict()
  _backward_hooks: OrderedDict()
  _is_full_backward_hook: None
  _forward_hooks: OrderedDict()
  _forward_hooks_with_kwargs: OrderedDict()
  _forward_pre_hooks: OrderedDict()
  _forward_pre_hooks_with_kwargs: OrderedDict()
  _state_dict_hooks: OrderedDict()
  _state_dict_pre_hooks: OrderedDict()
  _load_state_dict_pre_hooks: OrderedDict()
  _load_state_dict_post_hooks: OrderedDict()
  _modules: OrderedDict([('value_mean_std', RunningMeanStd()), ('running_mean_std', RunningMeanStd()), ('a2c_network', Network(
  (actor_cnn): Sequential()
  (critic_cnn): Sequential()
  (actor_mlp): Sequential(
    (0): Linear(in_features=23, out_features=256, bias=True)
    (1): ELU(alpha=1.0)
    (2): Linear(in_features=256, out_features=128, bias=True)
    (3): ELU(alpha=1.0)
    (4): Linear(in_features=128, out_features=64, bias=True)
    (5): ELU(alpha=1.0)
  )
  (critic_mlp): Sequential()
  (value): Linear(in_features=64, out_features=1, bias=True)
  (value_act): Identity()
  (mu): Linear(in_features=64, out_features=9, bias=True)
  (mu_act): Identity()
  (sigma_act): Identity()
))])
  obs_shape: (23,)
  normalize_value: True
  normalize_input: True
  value_size: 1

```


## Network
Retained the parts relevant to franka_cabinet.
A2CBuilder.Network is the a2c_network in the model above

```python
class A2CBuilder(NetworkBuilder):
    def __init__(self, **kwargs):
        NetworkBuilder.__init__(self)

    def load(self, params):
        self.params = params

    class Network(NetworkBuilder.BaseNetwork):
        def __init__(self, params, **kwargs):
            # ... trncated
            pass

        def forward(self, obs_dict):
            obs = obs_dict['obs']
            states = obs_dict.get('rnn_states', None)
            seq_length = obs_dict.get('seq_length', 1)
            dones = obs_dict.get('dones', None)
            bptt_len = obs_dict.get('bptt_len', 0)

            if self.separate:
              # ... truncated
              pass
            else:
                out = obs
                # out = self.actor_cnn(out)  # blank
                out = out.flatten(1)                

                if self.has_rnn:
                  # ... truncated
                  pass
                else:
                    out = self.actor_mlp(out)
                value = self.value_act(self.value(out))

                if self.is_continuous:
                    mu = self.mu_act(self.mu(out))
                    if self.fixed_sigma:
                        sigma = self.sigma_act(self.sigma)
                    else:
                        # sigma = self.sigma_act(self.sigma(out))
                        pass
                    return mu, mu*0 + sigma, value, states
                    
```