import torch
import random
from torch import nn
from torchvision.models import resnet101


class Attention(nn.Module):
    def __init__(self, encoder_dims, decoder_dims, hidden_dims):
        super(Attention, self).__init__()
        self.project_feature = nn.Linear(encoder_dims, hidden_dims)
        self.project_hidden = nn.Linear(decoder_dims, hidden_dims)
        self.project_embedding = nn.Linear(hidden_dims, 1)
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)

    def forward(self, feature, hidden):
        feature_embedding = self.project_feature(feature) # (batch_size, num_pixels, hidden_dims)
        hidden_embedding = self.project_hidden(hidden).unsqueeze(1) # (batch_size, 1, hidden_dims)
        embedding = self.relu(feature_embedding + hidden_embedding) # (batch_size, num_pixels, hidden_dims)
        weight = self.softmax(self.project_embedding(embedding)) # (batch_size, num_pixels, 1)
        output = (weight * feature).sum(dim=1) # (batch_size, encoder_dims)
        return output, weight.squeeze(2)


class Encoder(nn.Module):
    def __init__(self, target_size=14):
        super(Encoder, self).__init__()
        resnet = resnet101(weights='DEFAULT')
        modules = list(resnet.children())[:-2]
        self.resnet = nn.Sequential(*modules)
        self.resize = nn.AdaptiveAvgPool2d(target_size)
        self.configure()

    def configure(self, finetune=False):
        for parameter in self.resnet.parameters():
            parameter.requires_grad = False
        layers = list(self.resnet.children())[5:]
        for layer in layers:
            for parameter in layer.parameters():
                parameter.requires_grad = finetune

    def forward(self, input):
        output = self.resnet(input) # (batch_size, num_channels, image_size / 32, image_size / 32)
        output = self.resize(output) # (batch_size, num_channels, target_size, target_size)
        output = output.permute(0, 2, 3, 1) # (batch_size, target_size, target_size, num_channels)
        return output


class Decoder(nn.Module):
    def __init__(self, vocabulary_size, embedding_dims=256, encoder_dims=2048, decoder_dims=256, hidden_dims=256, dropout_rate=0.5):
        super(Decoder, self).__init__()
        self.vocabulary_size = vocabulary_size
        self.attention = Attention(encoder_dims, decoder_dims, hidden_dims)
        self.embedding = nn.Embedding(num_embeddings=vocabulary_size, embedding_dim=embedding_dims)
        self.dropout = nn.Dropout(dropout_rate)
        self.sigmoid = nn.Sigmoid()
        self.decode = nn.LSTMCell(embedding_dims + encoder_dims, decoder_dims)
        self.hidden = nn.Linear(encoder_dims, decoder_dims)
        self.cell = nn.Linear(encoder_dims, decoder_dims)
        self.gate = nn.Linear(decoder_dims, encoder_dims)
        self.head = nn.Linear(decoder_dims, vocabulary_size)
        self.configure()

    def configure(self):
        self.embedding.weight.data.uniform_(-0.1, 0.1)
        self.head.weight.data.uniform_(-0.1, 0.1)
        self.head.bias.data.fill_(0)

    def reset(self, image_feature):
        mean_feature = image_feature.mean(dim=1) # (batch_size, encoder_dims)
        hidden_state = self.hidden(mean_feature) # (batch_size, decoder_dims)
        cell_state = self.cell(mean_feature) # (batch_size, decoder_dims)
        return hidden_state, cell_state

    def forward(self, image_feature, caption_token, caption_length, sample_epsilon=1.0):
        image_feature = image_feature.flatten(start_dim=1, end_dim=2) # (batch_size, num_pixels, encoder_dims)
        batch_size, num_pixels, _ = image_feature.shape
        caption_length, sorted_index = caption_length.squeeze(1).sort(dim=0, descending=True) # (batch_size, ), (batch_size, )
        image_feature, caption_token = image_feature[sorted_index], caption_token[sorted_index] # (batch_size, num_pixels, encoder_dims), (batch_size, max_length)
        target_length = (caption_length - 1).tolist()
        max_length = max(target_length)

        hidden_state, cell_state = self.reset(image_feature) # (batch_size, decoder_dims), (batch_size, decoder_dims)
        caption_embedding = self.embedding(caption_token) # (batch_size, max_length, embedding_dims)
        caption_prediction = torch.zeros(batch_size, max_length, self.vocabulary_size).to(image_feature.device) # (batch_size, max_length, vocabulary_size)
        attention_weight = torch.zeros(batch_size, max_length, num_pixels).to(image_feature.device) # (batch_size, max_length, num_pixels)

        for step in range(max_length):
            block_size = sum([length > step for length in target_length])
            output, weight = self.attention(image_feature[:block_size], hidden_state[:block_size]) # (block_size, encoder_dims), (block_size, num_pixels)
            gate = self.sigmoid(self.gate(hidden_state[:block_size])) # (block_size, encoder_dims)
            output = gate * output # (block_size, encoder_dims)
            if step == 0 or random.random() < sample_epsilon:
                embedding = caption_embedding[:block_size, step] # (block_size, embedding_dims)
            else:
                embedding = self.embedding(last_token[:block_size]) # (block_size, embedding_dims)
            hidden_state, cell_state = self.decode(
                torch.cat([embedding, output], dim=1), # (block_size, embedding_dims + encoder_dims)
                (hidden_state[:block_size], cell_state[:block_size]) # (block_size, decoder_dims), (block_size, decoder_dims)
            ) # (block_size, decoder_dims), (block_size, decoder_dims)
            prediction = self.head(self.dropout(hidden_state)) # (block_size, vocabulary_size)
            last_token = prediction.argmax(dim=1) # (block_size, )
            caption_prediction[:block_size, step] = prediction
            attention_weight[:block_size, step] = weight

        return caption_prediction, caption_token, target_length, attention_weight, sorted_index

    def sample(self, image_feature, start_index, end_index=-1, max_length=40, return_weight=False, sample_method='beam', beam_size=5):
        batch_size, feature_size, _, _ = image_feature.shape
        sample_token = []
        sample_weight = []

        if sample_method == 'greedy':
            last_token = torch.LongTensor([[start_index]] * batch_size).to(image_feature.device) # (batch_size, 1)
            image_feature = image_feature.flatten(start_dim=1, end_dim=2) # (batch_size, num_pixels, encoder_dims)
            hidden_state, cell_state = self.reset(image_feature) # (batch_size, decoder_dims), (batch_size, decoder_dims)

            for step in range(max_length):
                embedding = self.embedding(last_token).squeeze(1) # (batch_size, embedding_dims)
                output, weight = self.attention(image_feature, hidden_state) # (batch_size, encoder_dims), (batch_size, num_pixels)
                weight = weight.view(batch_size, feature_size, feature_size).unsqueeze(1) # (batch_size, 1, feature_size, feature_size)
                gate = self.sigmoid(self.gate(hidden_state)) # (batch_size, encoder_dims)
                output = gate * output # (batch_size, encoder_dims)
                hidden_state, cell_state = self.decode(
                    torch.cat([embedding, output], dim=1), # (batch_size, embedding_dims + encoder_dims)
                    (hidden_state, cell_state) # (batch_size, decoder_dims), (batch_size, decoder_dims)
                ) # (batch_size, decoder_dims), (batch_size, decoder_dims)
                prediction = self.head(hidden_state) # (batch_size, vocabulary_size)

                next_token = prediction.argmax(dim=1) # (batch_size, )
                last_token = next_token.unsqueeze(1) # (batch_size, 1)
                sample_token.append(next_token)
                sample_weight.append(weight.squeeze(1))

            sample_token = torch.stack(sample_token, dim=1) # (batch_size, max_length)
            sample_weight = torch.stack(sample_weight, dim=1) # (batch_size, max_length, num_pixels)

        elif sample_method == 'beam':
            for instance in range(batch_size):
                sample_beam = []
                best_score = torch.zeros(beam_size).to(image_feature.device) # (beam_size, )
                last_token = torch.LongTensor([[start_index]] * beam_size).to(image_feature.device) # (beam_size, 1)
                feature_instance = image_feature[instance].unsqueeze(0) # (1, feature_size, feature_size, encoder_dims)
                feature_instance = feature_instance.repeat_interleave(beam_size, dim=0) # (beam_size, feature_size, feature_size, encoder_dims)
                feature_instance = feature_instance.flatten(start_dim=1, end_dim=2) # (beam_size, num_pixels, encoder_dims)
                hidden_state, cell_state = self.reset(feature_instance) # (beam_size, decoder_dims), (beam_size, decoder_dims)

                for step in range(max_length):
                    embedding = self.embedding(last_token).squeeze(1) # (beam_size, embedding_dims)
                    output, weight = self.attention(feature_instance, hidden_state) # (beam_size, encoder_dims), (beam_size, num_pixels)
                    weight = weight.view(beam_size, feature_size, feature_size).unsqueeze(1) # (beam_size, 1, feature_size, feature_size)
                    gate = self.sigmoid(self.gate(hidden_state)) # (beam_size, encoder_dims)
                    output = gate * output # (beam_size, encoder_dims)
                    hidden_state, cell_state = self.decode(
                        torch.cat([embedding, output], dim=1), # (beam_size, embedding_dims + encoder_dims)
                        (hidden_state, cell_state) # (beam_size, decoder_dims), (beam_size, decoder_dims)
                    ) # (beam_size, decoder_dims), (beam_size, decoder_dims)
                    prediction = self.head(hidden_state) # (beam_size, vocabulary_size)

                    current_score = best_score.unsqueeze(1) + torch.log_softmax(prediction, dim=1) # (beam_size, vocabulary_size)
                    if step == 0:
                        best_score, best_token = current_score[0].topk(beam_size, dim=0) # (beam_size, ), (beam_size, )
                    else:
                        best_score, best_token = current_score.flatten().topk(beam_size, dim=0) # (beam_size, ), (beam_size, )
                    last_beam = best_token // self.vocabulary_size # (beam_size, )
                    next_token = best_token % self.vocabulary_size # (beam_size, )
                    if step == 0:
                        sequence_instance = next_token.unsqueeze(1) # (beam_size, 1)
                    else:
                        sequence_instance = torch.cat([sequence_instance[last_beam], next_token.unsqueeze(1)], dim=1) # (beam_size, step + 1)
                    last_token = next_token.unsqueeze(1) # (beam_size, 1)

                    end_flag = (next_token == end_index) # (beam_size, )
                    if step == max_length - 1:
                        end_flag.fill_(1) # (beam_size, )
                    for beam in range(beam_size):
                        if end_flag[beam]:
                            final_beam = {'sequence': sequence_instance[beam].clone(), 'score': best_score[beam].item() / (step + 1)}
                            sample_beam.append(final_beam)
                    best_score[end_flag] -= 1000 # (beam_size, )

                sample_token.append(max(sample_beam, key=lambda x: x['score'])['sequence'])
                sample_weight.append(weight[0].squeeze(0))

            sample_token = torch.stack(sample_token, dim=0) # (batch_size, max_length)
            sample_weight = torch.stack(sample_weight, dim=0) # (batch_size, max_length, num_pixels)

        if return_weight:
            return sample_token, sample_weight
        else:
            return sample_token


class Captioner(nn.Module):
    def __init__(self, vocabulary_size, feature_size=14, embedding_dims=256, encoder_dims=2048, decoder_dims=256, hidden_dims=256, dropout_rate=0.5):
        super(Captioner, self).__init__()
        self.encoder = Encoder(target_size=feature_size)
        self.decoder = Decoder(vocabulary_size, embedding_dims, encoder_dims, decoder_dims, hidden_dims, dropout_rate)

    def forward(self, source_image, caption_token, caption_length, sample_epsilon=1.0):
        image_feature = self.encoder(source_image)
        decode_result = self.decoder(image_feature, caption_token, caption_length.unsqueeze(1), sample_epsilon)
        return decode_result

    def sample(self, source_image, start_index, end_index=-1, max_length=40, return_weight=False, sample_method='beam', beam_size=5):
        image_feature = self.encoder(source_image)
        sample_result = self.decoder.sample(image_feature, start_index, end_index, max_length, return_weight, sample_method, beam_size)
        return sample_result
