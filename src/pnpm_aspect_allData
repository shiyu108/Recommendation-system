# -*- coding: utf-8 -*-
import os
import tensorflow as tf
import numpy as np


class PNPM_aspect_allData():
    def __init__(self, args, num_user, num_item, num_aspect, MaxPerUserPos, MaxPerUserNeg, MaxPerItem, user_aspect_pos_padded, user_aspect_neg_padded,
                 item_aspect_padded, aspect_vectors):
        self.num_user = num_user
        self.num_item = num_item
        self.num_aspect = num_aspect
        self.MaxPerUserPos = MaxPerUserPos
        self.MaxPerUserNeg = MaxPerUserNeg
        self.MaxPerItem = MaxPerItem
        self.num_mf_factor = args.num_mf_factor
        self.num_aspect_factor = args.num_aspect_factor
        self.num_attention = args.num_attention
        self.args = args
        self.user_aspect_pos_padded = user_aspect_pos_padded  
        self.user_aspect_neg_padded = user_aspect_neg_padded
        self.item_aspect_padded = item_aspect_padded  
        self.aspect_vectors = aspect_vectors

    def _create_placeholders(self):
        self.user_input = tf.placeholder(dtype=tf.int32, shape=[None, ], name="user_input")
        self.item_p_input = tf.placeholder(dtype=tf.int32, shape=[None, ], name="item_p_input")
        self.item_n_input = tf.placeholder(dtype=tf.int32, shape=[None, ], name="item_n_input")
        self.dropout_keep = tf.placeholder(tf.float32, name="dropout_keep")


    def _create_variables(self):
        self.all_weights = {}
        PADDING_ID = 0
        num_row_lookup_table = self.num_aspect

        raw_mask_array = [[1.]] * PADDING_ID + [[0.]] + [[1.]] * (num_row_lookup_table - PADDING_ID - 1)
        
        self.all_weights['mask_lookup_table'] = tf.get_variable("mask_lookup_table",
                                                                initializer=raw_mask_array,
                                                                dtype=tf.float32,
                                                                trainable=False)
        self.all_weights['user_history_aspect_pos'] = tf.get_variable("user_history_aspect_pos",
                                                                  initializer=self.user_aspect_pos_padded,
                                                                  dtype=tf.int32,
                                                                  trainable=False)
        self.all_weights['user_history_aspect_neg'] = tf.get_variable("user_history_aspect_neg",
                                                                      initializer=self.user_aspect_neg_padded,
                                                                      dtype=tf.int32,
                                                                      trainable=False)
        self.all_weights['item_history_aspect'] = tf.get_variable("item_history_aspect",
                                                                  initializer=self.item_aspect_padded,
                                                                  dtype=tf.int32,
                                                                  trainable=False)
        #global interaction
        self.all_weights['user_embed'] = tf.get_variable('user_embed', shape=[self.num_user, self.num_mf_factor],
                                                         dtype=tf.float32,
                                                         initializer=tf.uniform_unit_scaling_initializer(factor=1.0))
        self.all_weights['item_embed'] = tf.get_variable('item_embed', shape=[self.num_item, self.num_mf_factor],
                                                         dtype=tf.float32,
                                                         initializer=tf.uniform_unit_scaling_initializer(factor=1.0))

        self.all_weights['W_out'] = tf.get_variable(name='W_out', dtype=tf.float32,
                                                    shape=[self.num_aspect_factor + self.num_mf_factor, 1],
                                                    initializer=tf.uniform_unit_scaling_initializer(factor=1.0))

        self.all_weights['aspect_embed'] = tf.get_variable('aspect_embed', dtype=tf.float32,
                                                           initializer=self.aspect_vectors, trainable=False)

    def _create_inference(self, item_input, is_reuse):
        with tf.name_scope('global_module'):
            u_emb = tf.nn.embedding_lookup(self.all_weights['user_embed'], self.user_input)
            v_emb = tf.nn.embedding_lookup(self.all_weights['item_embed'], item_input)

            mf_interact = tf.nn.dropout(tf.multiply(u_emb, v_emb), keep_prob=self.dropout_keep)

        with tf.name_scope('aspect_module'):
            u_hist_pos = tf.nn.embedding_lookup(self.all_weights['user_history_aspect_pos'], self.user_input, name='u_hist_pos') #12 aspects of user_input
            u_hist_neg = tf.nn.embedding_lookup(self.all_weights['user_history_aspect_neg'], self.user_input, name='u_hist_neg')
            v_hist = tf.nn.embedding_lookup(self.all_weights['item_history_aspect'], item_input, name='v_hist')

            u_hist_a_pos_embs = tf.nn.embedding_lookup(self.all_weights['aspect_embed'], u_hist_pos, name='u_hist_a_pos_embs')
            u_hist_a_neg_embs = tf.nn.embedding_lookup(self.all_weights['aspect_embed'], u_hist_neg, name='u_hist_a_neg_embs')
            v_hist_a_embs = tf.nn.embedding_lookup(self.all_weights['aspect_embed'], v_hist, name='v_hist_a_embs')



            u_hist_a_pos_embs = tf.layers.dense(u_hist_a_pos_embs, units=self.num_aspect_factor,
                                            name='aspect_embed_pos_trans',
                                            kernel_initializer=tf.uniform_unit_scaling_initializer(factor=1.0),
                                            use_bias=False,
                                            reuse=is_reuse)

            u_hist_a_neg_embs = tf.layers.dense(u_hist_a_neg_embs, units=self.num_aspect_factor,
                                                name='aspect_embed_neg_trans',
                                                kernel_initializer=tf.uniform_unit_scaling_initializer(factor=1.0),
                                                use_bias=False,
                                                reuse=is_reuse)

            v_hist_a_embs = tf.layers.dense(v_hist_a_embs, units=self.num_aspect_factor,
                                            name='aspect_embed_trans',
                                            kernel_initializer=tf.uniform_unit_scaling_initializer(factor=1.0),
                                            use_bias=False,
                                            reuse=is_reuse)

            user_mask_pos_padding = tf.nn.embedding_lookup(self.all_weights['mask_lookup_table'], u_hist_pos,
                                                       name='user_mask_pos_padding')
            user_mask_neg_padding = tf.nn.embedding_lookup(self.all_weights['mask_lookup_table'], u_hist_neg,
                                                           name='user_mask_neg_padding')
            item_mask_padding = tf.nn.embedding_lookup(self.all_weights['mask_lookup_table'], v_hist,
                                                       name='item_mask_padding')

            u_hist_a_pos_embs = tf.multiply(user_mask_pos_padding, u_hist_a_pos_embs, 'u_hist_a_pos_embs_masked')
            u_hist_a_neg_embs = tf.multiply(user_mask_neg_padding, u_hist_a_neg_embs, 'u_hist_a_neg_embs_masked')
            v_hist_a_embs = tf.multiply(item_mask_padding, v_hist_a_embs, 'v_hist_a_embs_masked')


            with tf.name_scope('aspect_interact'):
                u_hist_a_pos_embs_interact = tf.nn.l2_normalize(u_hist_a_pos_embs, dim=-1) #正则化
                u_hist_a_neg_embs_interact = tf.nn.l2_normalize(u_hist_a_neg_embs, dim=-1)  # 正则化
                v_hist_a_embs_interact = tf.nn.l2_normalize(v_hist_a_embs, dim=-1)


                u_aspect_pos_array_ = tf.expand_dims(u_hist_a_pos_embs_interact, 2)
                u_aspect_neg_array_ = tf.expand_dims(u_hist_a_neg_embs_interact, 2)
                v_aspect_array_ = tf.expand_dims(v_hist_a_embs_interact, 1)
                #self.user_pos_shape = u_aspect_pos_array_
                #self.user_neg_shape = u_aspect_neg_array_
                #self.item_shape = v_aspect_array_

                interact_pos = tf.multiply(u_aspect_pos_array_, v_aspect_array_)
                interact_neg = tf.multiply(u_aspect_neg_array_, v_aspect_array_)
                #self.mul_result = interact

            with tf.name_scope('aspect_level_attention'):
                att_l2_1_pos = tf.layers.dense(interact_pos, units=1, name='att_l2_1_pos', reuse=is_reuse)
                att_l2_pos = tf.nn.softmax(att_l2_1_pos, dim=2)

                att_l2_1_neg = tf.layers.dense(interact_neg, units=1, name='att_l2_1_neg', reuse=is_reuse)
                att_l2_neg = tf.nn.softmax(att_l2_1_neg, dim=2)

            with tf.name_scope("user_level_attention"):
                v_a_emb_pos = tf.tile(tf.reduce_sum(v_hist_a_embs_interact, axis=1, keep_dims=True),
                                  [1, self.MaxPerUserPos, 1])
                input_att_l1_pos = v_a_emb_pos * u_hist_a_pos_embs_interact
                att_l1_1_pos = tf.layers.dense(input_att_l1_pos, units=1, name='att_l1_1_pos', reuse=is_reuse)
                att_l1_pos = tf.nn.softmax(att_l1_1_pos, dim=1)

                v_a_emb_neg = tf.tile(tf.reduce_sum(v_hist_a_embs_interact, axis=1, keep_dims=True),
                                  [1, self.MaxPerUserNeg, 1])
                input_att_l1_neg = v_a_emb_neg * u_hist_a_neg_embs_interact
                att_l1_1_neg = tf.layers.dense(input_att_l1_neg, units=1, name='att_l1_1_neg', reuse=is_reuse)
                att_l1_neg = tf.nn.softmax(att_l1_1_neg, dim=1)

        with tf.name_scope('attach_attention'):
            weighted_interact_l2_pos = tf.reduce_sum(tf.multiply(att_l2_pos, interact_pos), axis=2)
            aspect_interact_pos = tf.reduce_sum(tf.multiply(att_l1_pos, weighted_interact_l2_pos), axis=1)
            aspect_interact_pos = tf.nn.dropout(aspect_interact_pos, self.dropout_keep)

            weighted_interact_l2_neg = tf.reduce_sum(tf.multiply(att_l2_neg, interact_neg), axis=2)
            aspect_interact_neg = tf.reduce_sum(tf.multiply(att_l1_neg, weighted_interact_l2_neg), axis=1)
            aspect_interact_neg = tf.nn.dropout(aspect_interact_neg, self.dropout_keep)
            #aspect_interact=tf.subtract(aspect_interact_pos,aspect_interact_neg)

        with tf.name_scope('concatenate'):
            aspect_interact = tf.subtract(aspect_interact_pos, aspect_interact_neg)
            interact_vector = tf.concat([mf_interact, aspect_interact], axis=-1)
            #interact_vector = tf.concat([aspect_interact_pos, aspect_interact_neg], axis=-1)

        with tf.name_scope('prediction'):
            rating_preds = tf.matmul(interact_vector, self.all_weights['W_out'], name='prediction')

        return rating_preds

    def _create_loss(self):
        with tf.name_scope('P_infer'):
            self.rating_preds_pos = self._create_inference(self.item_p_input, is_reuse=None)
        with tf.name_scope('N_infer'):
            self.rating_preds_neg = self._create_inference(self.item_n_input, is_reuse=True)
        with tf.name_scope('loss'):
            x = self.rating_preds_pos - self.rating_preds_neg
            x = tf.clip_by_value(x, -80.0, 1e8)
            self.bprloss = -tf.reduce_mean(tf.log(tf.sigmoid(x)), name='bpr_loss')
            tf.add_to_collection("losses", self.bprloss)

            if self.args.is_out_l2:
                tf.add_to_collection("losses",
                                     self.args.lamda_out_l2 * (tf.reduce_mean(self.all_weights['W_out'] ** 2)))
            if self.args.is_l2_regular:
                tf.add_to_collection("losses",
                                     self.args.lamda_l2 * (tf.reduce_mean(self.all_weights['user_embed'] ** 2)))
                tf.add_to_collection("losses",
                                     self.args.lamda_l2 * (tf.reduce_mean(self.all_weights['item_embed'] ** 2)))
            self.loss = tf.add_n(tf.get_collection("losses"))

    def _create_train_op(self):
        with tf.name_scope('train'):
            self.global_step = tf.Variable(0, name='global_step', trainable=False)
        if self.args.optimizer_type == 1:
            self.opt = tf.train.AdamOptimizer(learning_rate=self.args.learning_rate, beta1=0.9, beta2=0.999,
                                              epsilon=1e-8)
            self.grads = self.opt.compute_gradients(self.loss, tf.trainable_variables())
            self.train_op = self.opt.apply_gradients(self.grads, global_step=self.global_step, name='train')

        elif self.args.optimizer_type == 2:
            self.opt = tf.train.AdagradOptimizer(learning_rate=self.args.learning_rate, initial_accumulator_value=1e-8)
            self.grads = self.opt.compute_gradients(self.loss, tf.trainable_variables())
            self.train_op = self.opt.apply_gradients(self.grads, global_step=self.global_step, name='train')

        elif self.args.optimizer_type == 3:
            self.opt = tf.train.GradientDescentOptimizer(learning_rate=self.args.learning_rate)
            self.grads = self.opt.compute_gradients(self.loss, tf.trainable_variables())
            self.train_op = self.opt.apply_gradients(self.grads, global_step=self.global_step, name='train')

        elif self.args.optimizer_type == 4:
            self.opt = tf.train.MomentumOptimizer(learning_rate=self.args.learning_rate, momentum=0.95)
            self.grads = self.opt.compute_gradients(self.loss, tf.trainable_variables())
            self.train_op = self.opt.apply_gradients(self.grads, global_step=self.global_step, name='train')

    def build_graph(self):
        self._create_placeholders()
        self._create_variables()
        self._create_loss()
        self._create_train_op()
