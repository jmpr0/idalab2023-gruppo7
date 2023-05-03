import torch
from torch import nn
from argparse import ArgumentParser


class Fce(nn.Module):

    lstm_layers = 1
    unrolling_steps = 2
    apply_fce = False

    def __init__(self, lstm_input_size, **kwargs):
        """
        **Arguments**
        * **apply_fce** (bool, *optional*, default='False') - Whether or not to us fully conditional embeddings.
        * **lstm_layers**: Number of LSTM layers in the bidrectional LSTM g that embeds the support set.
        * **unrolling_steps**: Number of unrolling steps to run the Attention LSTM.
        """
        super().__init__()
        self.apply_fce = kwargs.get(
            "apply_fce", Fce.apply_fce
        )
        self.lstm_layers = kwargs.get(
            "lstm_layers", Fce.lstm_layers
        )
        self.unrolling_steps = kwargs.get(
            "unrolling_steps", Fce.unrolling_steps
        )
        self._device = kwargs.get(
            "device", "cpu"
        )
        self.lstm_input_size = lstm_input_size
        
        g = BidrectionalLSTM(
            self.lstm_input_size,
            self.lstm_layers
        ).to(self._device)
        f = AttentionLSTM(
            self.lstm_input_size, 
            unrolling_steps=self.unrolling_steps,
            device = self._device
        ).to(self._device)
        self.units = nn.ModuleDict({
            'g' : g,
            'f' : f
        })
        print('Using FCE') if self.apply_fce else print('No FCE')
        
    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = ArgumentParser(
            parents=[parent_parser],
            add_help=True,
            conflict_handler="resolve",
        )
        parser.add_argument("--apply_fce", action='store_true', default=Fce.apply_fce)
        parser.add_argument("--lstm_layers", type=int, default=Fce.lstm_layers)
        parser.add_argument("--unrolling_steps", type=int, default=Fce.unrolling_steps)
        return parser

    # FCE does not work for mm networks
    def forward(self, support, query):
        support, _, _ = self.units['g'](support.unsqueeze(1))
        support = support.squeeze(1)
        query = self.units['f'](support, query)
        return support, query


class BidrectionalLSTM(nn.Module):
    def __init__(self, size: int, layers: int):
        """
        Bidirectional LSTM used to generate fully conditional embeddings (FCE) of the support set as described
        in the Matching Networks paper.

        **Arguments**
            size: Size of input and hidden layers. These are constrained to be the same in order to implement the skip
                connection described in Appendix A.2
            layers: Number of LSTM layers
        """
        super(BidrectionalLSTM, self).__init__()
        self.num_layers = layers
        self.batch_size = 1
        # Force input size and hidden size to be the same in order to implement
        # the skip connection as described in Appendix A.1 and A.2 of Matching Networks
        self.lstm = nn.LSTM(
            input_size=size,
            num_layers=layers,
            hidden_size=size,
            bidirectional=True
        )

    def forward(self, inputs):
        # Give None as initial state and Pytorch LSTM creates initial hidden states
        output, (hn, cn) = self.lstm(inputs, None)

        forward_output = output[:, :, :self.lstm.hidden_size]
        backward_output = output[:, :, self.lstm.hidden_size:]

        # g(x_i, S) = h_forward_i + h_backward_i + g'(x_i) as written in Appendix A.2
        # AKA A skip connection between inputs and outputs is used
        output = forward_output + backward_output + inputs
        return output, hn, cn


class AttentionLSTM(nn.Module):
    def __init__(self, size: int, unrolling_steps: int, device: str):
        """
        Attentional LSTM used to generate fully conditional embeddings (FCE) of the query set as described
        in the Matching Networks paper.

        **Arguments**
            size: Size of input and hidden layers. These are constrained to be the same in order to implement the skip
                connection described in Appendix A.2
            unrolling_steps: Number of steps of attention over the support set to compute. Analogous to number of
                layers in a regular LSTM
        """
        super(AttentionLSTM, self).__init__()
        self.unrolling_steps = unrolling_steps
        self._device = device  
        self.lstm_cell = nn.LSTMCell(
            input_size=size,
            hidden_size=size
        ) 

    def forward(self, support, queries):
        # Get embedding dimension, d
        if support.shape[-1] != queries.shape[-1]:
            raise(ValueError("Support and query set have different embedding dimension!"))

        batch_size = queries.shape[0]
        embedding_dim = queries.shape[1]

        h_hat = torch.zeros_like(queries).to(self._device)
        c = torch.zeros(batch_size, embedding_dim).to(self._device)

        for k in range(self.unrolling_steps):
            # Calculate hidden state cf. equation (4) of appendix A.2
            h = h_hat + queries
            
            # Calculate softmax attentions between hidden states and support set embeddings
            # cf. equation (6) of appendix A.2
            attentions = torch.mm(h, support.t())
            attentions = attentions.softmax(dim=1)

            # Calculate readouts from support set embeddings cf. equation (5)
            readout = torch.mm(attentions, support)

            # Run LSTM cell cf. equation (3)
            # h_hat, c = self.lstm_cell(queries, (torch.cat([h, readout], dim=1), c))
            h_hat, c = self.lstm_cell(queries, (h + readout, c))

        h = h_hat + queries

        return h