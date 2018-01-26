from src.models.classifier_models import *

class UschemaModel(ClassifierModel):

    def __init__(self, ep_vocab_size, entity_vocab_size, kb_vocab_size, token_vocab_size, position_vocab_size,
                 ner_label_vocab_size, embeddings, entity_embeddings, string_int_maps, FLAGS):
        self.model_type = 'classifier'
        self._lr = FLAGS.lr
        self._embed_dim = FLAGS.embed_dim
        self.inner_embed_dim = FLAGS.embed_dim
        self._token_dim = FLAGS.token_dim
        self._lstm_dim = FLAGS.lstm_dim
        self._position_dim = FLAGS.position_dim
        self._kb_size = kb_vocab_size
        self._token_size = token_vocab_size
        self._position_size = position_vocab_size
        self._ep_vocab_size = ep_vocab_size
        self._entity_vocab_size = entity_vocab_size
        self._num_labels = FLAGS.num_classes
        self._peephole = FLAGS.use_peephole
        self.string_int_maps = string_int_maps
        self.FLAGS = FLAGS

        self._epsilon = tf.constant(0.00001, name='epsilon')
        self.global_step = tf.Variable(0, name='global_step', trainable=False)

        self.non_linear = tf.nn.tanh if FLAGS.use_tanh else tf.identity
        self.max_pool = FLAGS.max_pool
        self.bidirectional = FLAGS.bidirectional
        self.margin = FLAGS.margin
        self.verbose = FLAGS.verbose
        self.freeze = FLAGS.freeze
        self.mlp = FLAGS.mlp
        self.entity_index = 1

        # set up placeholders
        self.label_batch = tf.placeholder_with_default([0], [None], name='label_batch')
        self.ner_label_batch = tf.placeholder_with_default([[0]], [None, None], name='ner_label_batch')
        self.kb_batch = tf.placeholder_with_default([0], [None], name='kb_batch')
        self.e1_batch = tf.placeholder_with_default([0], [None], name='e1_batch')
        self.e2_batch = tf.placeholder_with_default([0], [None], name='e2_batch')
        self.ep_batch = tf.placeholder_with_default([0], [None], name='ep_batch')
        self.text_batch = tf.placeholder_with_default([[0]], [None, None], name='text_batch')
        self.e1_dist_batch = tf.placeholder_with_default([[0]], [None, None], name='e1_dist_batch')
        self.e2_dist_batch = tf.placeholder_with_default([[0]], [None, None], name='e2_dist_batch')
        self.ep_dist_batch = tf.placeholder_with_default([[[0.0]]], [None, None, None], name='ep_dist_batch')

        self.pos_encode_batch = tf.placeholder_with_default([[0]], [None, None], name='pos_encode_batch')
        self.seq_len_batch = tf.placeholder_with_default([0], [None], name='seq_len_batch')
        self.loss_weight = tf.placeholder_with_default(1.0, [], name='loss_weight')
        self.example_loss_weights = tf.placeholder_with_default([1.0], [None], name='example_loss_weights')

        self.word_dropout_keep = tf.placeholder_with_default(1.0, [], name='word_keep_prob')
        self.lstm_dropout_keep = tf.placeholder_with_default(1.0, [], name='lstm_keep_prob')
        self.final_dropout_keep = tf.placeholder_with_default(1.0, [], name='final_keep_prob')
        self.noise_weight = tf.placeholder_with_default(0.0, [], name='noise_weight')
        self.k_losses = tf.placeholder_with_default(0, [], name='k_losses')
        self.text_update = tf.placeholder_with_default(True, [], name='text_update')

        # initialize embedding tables
        with tf.variable_scope('noise_classifier'):
            if embeddings is None:
                self.token_embeddings = tf.get_variable(name='token_embeddings',
                                                        shape=[self._token_size, self._token_dim],
                                                        initializer=tf.contrib.layers.xavier_initializer(),
                                                        trainable=(not FLAGS.freeze)
                                                        )
            else:
                self.token_embeddings = tf.get_variable(name='token_embeddings',
                                                        initializer=embeddings.astype('float32'),
                                                        # trainable=(not freeze)
                                                        )
            self.token_embeddings = tf.concat((tf.zeros(shape=[1, FLAGS.token_dim]),
                                               self.token_embeddings[1:, :]), 0)

            self.position_embeddings = tf.get_variable(name='position_embeddings',
                                                       shape=[self._position_size, self._position_dim],
                                                       initializer=tf.contrib.layers.xavier_initializer()
                                                       ) if self._position_dim > 0 else tf.no_op()

            self.entity_embeddings = tf.get_variable(name='entity_embeddings',
                                                     shape=[self._ep_vocab_size, self._embed_dim],
                                                     initializer=tf.contrib.layers.xavier_initializer())

            self.kb_embeddings = tf.get_variable(name='kb_embeddings',
                                                     shape=[self._kb_size, self._embed_dim],
                                                     initializer=tf.contrib.layers.xavier_initializer())

            # MLP for scoring encoded sentence
            self.w_1 = tf.get_variable(name='w_1', shape=[self.inner_embed_dim, self._num_labels],
                                       initializer=tf.contrib.layers.xavier_initializer())
            self.b_1 = tf.get_variable(name='b_1', initializer=tf.constant(0.0001, shape=[self._num_labels]))

            if 'transform' in FLAGS.text_encoder:
                if 'cnn' in FLAGS.text_encoder and 'only' in FLAGS.text_encoder:
                    text_encoder_type = CNNAllPairs
                else:
                    text_encoder_type = Transformer
            elif 'glu' in FLAGS.text_encoder:
                text_encoder_type = GLUAllPairs
            else:
                print('%s is not a valid text encoder' % FLAGS.text_encoder)
                sys.exit(1)
            filter_width = int(FLAGS.text_encoder.split('_')[-1]) if FLAGS.text_encoder.split('_')[-1].isdigit() else 3
            self.text_encoder = text_encoder_type(self.text_batch, self.e1_dist_batch, self.e2_dist_batch, self.ep_dist_batch,
                                                  self.seq_len_batch,
                                                  self._lstm_dim, self._embed_dim, self._position_dim, self._token_dim,
                                                  self.bidirectional, self._peephole, self.max_pool,
                                                  self.word_dropout_keep, self.lstm_dropout_keep, self.final_dropout_keep,
                                                  layer_str=FLAGS.layer_str, pos_encode_batch=self.pos_encode_batch,
                                                  filterwidth=filter_width, block_repeats=FLAGS.block_repeats,
                                                  filter_pad=FLAGS.filter_pad, string_int_maps=string_int_maps,
                                                  entity_index=self.entity_index,
                                                    # entity_index=position_vocab_size/2,
                                                  e1_batch=self.e1_batch, e2_batch=self.e2_batch,
                                                  entity_embeddings=entity_embeddings, entity_vocab_size=entity_vocab_size,
                                                  num_labels=FLAGS.num_classes, project_inputs=FLAGS.freeze
                                                  )

            # encode the batch of sentences into b x d matrix
            token_attention = None  # tf.squeeze(self.get_ep_embedding(), [1])
            self.attention_vector = tf.nn.embedding_lookup(tf.transpose(self.w_1), tf.ones_like(self.e1_batch))
            encoded_text_list = self.text_encoder.embed_text(self.token_embeddings, self.position_embeddings,
                                                             self.attention_vector, token_attention=token_attention,
                                                             return_tokens=True)

            predictions_list = [self.add_bias(pred) for pred in encoded_text_list]
            self.predictions = encoded_text_list[-1]
            self.probs = self.get_probs()

            self.sentence_embeddings = self.pool_sentence(predictions_list[-1])

            self.rel_embeddings = tf.nn.embedding_lookup(self.kb_embeddings, self.kb_batch)

            self.loss = self.calculate_loss(encoded_text_list, predictions_list, self.label_batch,
                                            FLAGS.l2_weight, FLAGS.dropout_loss_weight, [])

            self.accuracy = self.calculate_accuracy(self.predictions, self.label_batch)
            self.text_variance = self.negative_prob(self.predictions)
            self.text_kb = self.positive_prob(self.predictions)
            self.kb_ep_score = self.compare_kb_ep()

            self.ner_predictions, self.ner_loss = self.ner(ner_labels=ner_label_vocab_size)

    def compare_kb_text(self):
        '''
        :return: cosine similarity between a batch of kb relations and a batch of text relations
        '''
        kb_normalized = tf.expand_dims(tf.contrib.layers.unit_norm(self.rel_embeddings, 1, .00001), 1)
        text_normalized = tf.expand_dims(tf.contrib.layers.unit_norm(self.sentence_embeddings, 1, .00001), 2)
        scores = tf.matmul(kb_normalized, text_normalized)
        return scores

    def compare_kb_ep(self, normalize=False):
        '''
        :return: cosine similarity between a batch of kb relations and a batch of ep vectors
        '''
        ep_embeddings = tf.nn.embedding_lookup(self.entity_embeddings, self.ep_batch)
        if normalize:
            kb_normalized = tf.expand_dims(tf.contrib.layers.unit_norm(self.rel_embeddings, 1, .00001), 1)
            ep_normalized = tf.expand_dims(tf.contrib.layers.unit_norm(ep_embeddings, 1, .00001), 2)
        else:
            kb_normalized = tf.expand_dims(self.rel_embeddings, 1)
            ep_normalized = tf.expand_dims(ep_embeddings, 2)
        scores = tf.matmul(kb_normalized, ep_normalized)
        return scores


    def hinge_loss(self, pos_predictions, neg_predictions, margin=1.0):
        '''
        compute hinge loss between positive and negative triple,
        '''
        err = tf.nn.relu(neg_predictions - pos_predictions + tf.constant(margin))
        return err


    def pool_sentence(self, token_embeddings):
        token_embeddings = tf.layers.dense(tf.layers.dense(token_embeddings, self._embed_dim, activation=tf.nn.relu), self._embed_dim)
        token_embeddings = tf.nn.dropout(token_embeddings, self.final_dropout_keep)

        ones = tf.cast(tf.ones_like(self.text_batch), tf.float32)
        zeros = tf.cast(tf.zeros_like(self.text_batch), tf.float32)
        # non pad tokens
        token_mask = tf.where(tf.not_equal(self.text_batch, 0), ones, zeros)
        token_mask = tf.cast(tf.expand_dims(token_mask, 2), tf.float32)

        token_embeddings = tf.multiply(token_embeddings, token_mask)
        # instead of setting pad tokens to 0, set them to very -inf
        token_mask = tf.where(tf.equal(token_mask, 0),
                              np.NINF*tf.ones_like(token_mask),
                              tf.zeros_like(token_mask))
        token_embeddings = token_embeddings + token_mask

        sentence_embeddings = tf.reduce_max(token_embeddings, 1)

        return sentence_embeddings

    def calculate_loss(self, encoded_text_list, logits_list, labels, l2_weight,
                       dropout_weight=0.0, no_drop_output_list=None, label_smoothing=.0,
                       non_linear=tf.identity):
        loss = self.calculate_hinge_loss() if self.FLAGS.loss_type == 'hinge' else self.calcuate_sampled_softmax_loss()
        return self.loss_weight * loss

    def calculate_hinge_loss(self, non_linear=tf.identity):
        neg_ep_batch = tf.random_uniform(tf.shape(self.ep_batch), minval=0, maxval=self._ep_vocab_size, dtype=tf.int32,
                                         seed=None, name=None)
        pos_ep_embeddings = tf.expand_dims(tf.nn.embedding_lookup(self.entity_embeddings, self.ep_batch), 1)
        neg_ep_embeddings = tf.expand_dims(tf.nn.embedding_lookup(self.entity_embeddings, neg_ep_batch), 1)

        relation_embedding = tf.cond(self.text_update,
                                     lambda: self.sentence_embeddings,
                                     lambda: self.rel_embeddings)
        relation_embedding = non_linear(tf.expand_dims(relation_embedding, 2))

        pos_predictions = tf.matmul(pos_ep_embeddings, relation_embedding)
        neg_predictions_1 = tf.matmul(neg_ep_embeddings, relation_embedding)

        # neg_predictions_1 = tf.Print(neg_predictions_1, [self.text_update,
        #                                                  tf.reduce_mean(relation_embedding),
        #                                                  tf.reduce_mean(pos_ep_embeddings),
        #                                                  tf.reduce_mean(neg_ep_embeddings),
        #                                                  tf.reduce_mean(pos_predictions),
        #                                                  tf.reduce_mean(neg_predictions_1)], message='stats')

        return tf.reduce_mean(self.hinge_loss(pos_predictions, neg_predictions_1))

    def calcuate_sampled_softmax_loss(self, non_linear=tf.identity):

        samples = 100

        zero_bias = tf.constant(0.0, shape=[self._ep_vocab_size], dtype=tf.float32)

        relation_embedding = tf.cond(self.text_update,
                                     lambda: self.sentence_embeddings,
                                     lambda: self.rel_embeddings)
        # relation_embedding = non_linear(tf.expand_dims(relation_embedding, 2))

        labels = tf.expand_dims(tf.cast(self.ep_batch, tf.float32), 1)
        entity_loss = tf.nn.sampled_softmax_loss(self.entity_embeddings, zero_bias,
                                                 labels=labels, inputs=relation_embedding,
                                                 num_sampled=samples, num_classes=self._ep_vocab_size)
        loss = tf.reduce_mean(entity_loss)

        return loss


