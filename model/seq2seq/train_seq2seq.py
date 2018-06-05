# 训练一个使用所有特征的模型
import tkinter
import os
import pandas as pd
import numpy as np 
import seaborn as sns
import tensorflow as tf
import keras.backend.tensorflow_backend as KTF


from utils.metrics import SMAPE_on_dataset_v1
from model.seq2seq.seq2seq_model import build_graph
from model.seq2seq.seq2seq_data_util import get_training_statistics, generate_dev_set, generate_aggr_set, generate_X_test_set
from model.seq2seq.seq2seq_data_util import generate_training_list, oversampling_training_list, generate_training_batch
from model.seq2seq.seq2seq_data_util import pad, concate_aq_meo
from model.model_data_util import daily_statistics, expand_final_preds_dim, expand_day_trend,scale_day_trend
# Informations
from utils.information import bj_station_list, bj_X_aq_list, bj_y_aq_list
from utils.information import ld_station_list, ld_X_aq_list, ld_y_aq_list, ld_X_meo_list
bj_X_meo_list = ["temperature","pressure","humidity","direction","speed"]




def train_model(city='bj', 
                pre_days=5, 
                gap=0, 
                loss_function="L2", 
                total_iteractions=200,
                use_day_model=True,
                use_norm_data=True,
                generate_mean=False,
                generate_range=False,
                loss_weights=False,
                predict_one_day=False) :
    '''
    city='bj' or 'ld' : 针对某个城市的数据进行训练
    pre_days : 使用 pre_days 天数的数据进行预测
    gap : 0,12,24
        0 : 当天 23点以后进行的模型训练
        12 : 当天中午进行的模型训练
        24 : 不使用当天数据进行的训练
    loss_function : 使用不同的损失函数
    '''
    if city=="bj":
        station_list = bj_station_list
        X_aq_list = bj_X_aq_list
        y_aq_list = bj_y_aq_list
        X_meo_list = bj_X_meo_list
    elif city=="ld":
        station_list = ld_station_list
        X_aq_list = ld_X_aq_list
        y_aq_list = ld_y_aq_list
        X_meo_list = ld_X_meo_list     

    if use_day_model :
        input_seq_len = pre_days - gap
        if predict_one_day :
            output_seq_len = 1
        else :
            output_seq_len = 2
    else :
        input_seq_len = pre_days * 24 - gap
        if predict_one_day :
            output_seq_len = 24
        else :
            output_seq_len = 48
    
    if generate_mean and generate_range :
        input_dim = (len(station_list) * (len(X_aq_list) + len(X_meo_list))) * 2
        output_dim = len(station_list) * len(y_aq_list) * 2
    else :
        input_dim = len(station_list) * (len(X_aq_list) + len(X_meo_list))
        output_dim = len(station_list) * len(y_aq_list) 
    
    print(input_dim, output_dim)


    lambda_l2_reg = 0.003
    learning_rate = 1e-3 

    num_training_examples = 500
    use_day = True
    batch_size = 128
    hidden_dim = 512
    num_stacked_layers = 3
    GRADIENT_CLIPPING = 2.5
    total_iteractions = total_iteractions
    KEEP_RATE = 0.5

    output_features = []
    for station in station_list : 
        for aq_feature in y_aq_list :
            output_features.append(station + "_" + aq_feature)
    output_features.sort()

    

    # step 1 : generate data
    if use_norm_data :
        statistics = get_training_statistics(city)  # 统计量值
    
    if use_day_model :

        day_trend = daily_statistics(city=city,
                                     station_list=station_list,
                                     X_aq_list=X_aq_list,
                                     y_aq_list=y_aq_list,
                                     X_meo_list=X_meo_list,)  # shape of (24, features)
        print("is day_trend NAN ? ", np.isnan(day_trend).any().any())

        if generate_range :
            day_range_statistics = daily_range_statistics(city="bj",
                                                          station_list=station_list,
                                                          X_aq_list=X_aq_list,
                                                          y_aq_list=y_aq_list,
                                                          X_meo_list=X_meo_list)


    # training data
    X_df_list, y_df_list = generate_training_list(city=city, 
                                                  station_list=station_list, 
                                                  X_aq_list=X_aq_list, 
                                                  y_aq_list=y_aq_list, 
                                                  X_meo_list=X_meo_list, 
                                                  use_day=use_day, 
                                                  pre_days=pre_days, 
                                                  num_training_examples=num_training_examples, 
                                                  gap=gap,
                                                  use_norm_data=use_norm_data,
                                                  predict_one_day=predict_one_day)

    X_df_list, y_df_list = oversampling_training_list(X_df_list, 
                                                       y_df_list, 
                                                       oversample_part=0.4, 
                                                       repeats=10)


    X_predict = generate_X_test_set(city=city,
                                 station_list=station_list,
                                 X_aq_list=X_aq_list,
                                 X_meo_list=X_meo_list,
                                 pre_days=pre_days,
                                 gap=gap,
                                 use_day_model=use_day_model,
                                 use_norm_data=use_norm_data,
                                 generate_mean=generate_mean,
                                 generate_range=generate_range)


    if predict_one_day :

        dev_x, dev_y, dev_x_day_one, X_feature_filters, y_feature_filters = generate_dev_set(city=city,
                                                                                            station_list=station_list,
                                                                                            X_aq_list=X_aq_list, 
                                                                                            y_aq_list=y_aq_list, 
                                                                                            X_meo_list=X_meo_list,
                                                                                            pre_days=pre_days,
                                                                                            gap=gap,
                                                                                            use_day_model=use_day_model,
                                                                                            use_norm_data=use_norm_data,
                                                                                            generate_mean=generate_mean,
                                                                                            generate_range=generate_range,
                                                                                            for_aggr=False,
                                                                                            predict_one_day=False)
        print(dev_x.shape, dev_y.shape, dev_x_day_one.shape, len(X_feature_filters), len(y_feature_filters))

        aggr_x, aggr_y, aggr_x_day_one = generate_aggr_set(city=city, 
                                                           station_list=station_list, 
                                                           X_aq_list=X_aq_list, 
                                                           y_aq_list=y_aq_list, 
                                                           X_meo_list=X_meo_list, 
                                                           pre_days=pre_days, 
                                                           gap=gap,
                                                           use_day_model=use_day_model,
                                                           use_norm_data=use_norm_data,
                                                           generate_mean=generate_mean,
                                                           generate_range=generate_range,
                                                           predict_one_day=False)

    else :      
        dev_x, dev_y = generate_dev_set(city=city,
                                        station_list=station_list,
                                        X_aq_list=X_aq_list, 
                                        y_aq_list=y_aq_list, 
                                        X_meo_list=X_meo_list,
                                        pre_days=pre_days,
                                        gap=gap,
                                        use_day_model=use_day_model,
                                        use_norm_data=use_norm_data,
                                        generate_mean=generate_mean,
                                        generate_range=generate_range,
                                        for_aggr=False,
                                        predict_one_day=False)

        aggr_x, aggr_y = generate_aggr_set(city=city, 
                                           station_list=station_list, 
                                           X_aq_list=X_aq_list, 
                                           y_aq_list=y_aq_list, 
                                           X_meo_list=X_meo_list, 
                                           pre_days=pre_days, 
                                           gap=gap,
                                           use_day_model=use_day_model,
                                           use_norm_data=use_norm_data,
                                           generate_mean=generate_mean,
                                           generate_range=generate_range,
                                           predict_one_day=predict_one_day)


    if use_day_model :
        # if use day model, unpack ys
        dev_y_mean, dev_y = dev_y
        aggr_y_mean, aggr_y = aggr_y
    
    if use_norm_data :
        _, _, dev_y_original = SMAPE_on_dataset_v1(dev_y, dev_y, output_features, statistics, 1) 
        _, _, aggr_y_original = SMAPE_on_dataset_v1(aggr_y, aggr_y, output_features, statistics, 1) 
    else :
        dev_y_original = dev_y
        aggr_y_original = aggr_y
    
    print("Nan in dev_x",  np.isnan(dev_x).any().any())
    print("Nan in dev_y",  np.isnan(dev_y).any().any())



    # step 2 : train the model 
    # Define training model
    rnn_model = build_graph(feed_previous=False, 
                            input_seq_len=input_seq_len, 
                            output_seq_len=output_seq_len, 
                            hidden_dim=hidden_dim, 
                            input_dim=input_dim, 
                            output_dim=output_dim, 
                            num_stacked_layers=num_stacked_layers, 
                            learning_rate=learning_rate,
                            lambda_l2_reg=lambda_l2_reg,
                            GRADIENT_CLIPPING=GRADIENT_CLIPPING,
                            loss_function=loss_function,
                            use_norm_data=use_norm_data,
                            loss_weights=loss_weights)


    # training process
    train_losses = []
    val_losses = []
    saved_iteractions = []

    saver = tf.train.Saver()
    init = tf.global_variables_initializer()
    with tf.Session() as sess:

        sess.run(init)
        losses = []
        print("Training losses: ")
        for i in range(total_iteractions):


            batch_input, batch_output = generate_training_batch(X_df_list, 
                                                                y_df_list,
                                                                batch_size=batch_size,
                                                                use_day_model=use_day_model,
                                                                generate_mean=generate_mean,
                                                                generate_range=generate_range)




            if use_day_model : 
                # if use day model, unpack ys
                batch_output, _ = batch_output
            
            if use_norm_data and loss_weights:
                _, _, batch_output_original = SMAPE_on_dataset_v1(batch_output, batch_output, output_features, statistics, 1)
                batch_output_original  += 1e-3  # for numerical stability

            print(batch_input.shape, batch_output.shape)

            feed_dict = {rnn_model['enc_inp'][t]: batch_input[:,t,:] for t in range(input_seq_len)}
            feed_dict.update({rnn_model['target_seq'][t]: batch_output[:,t,:] for t in range(output_seq_len)})
            
            if use_norm_data and loss_weights :
                feed_dict.update({rnn_model['target_seq_original'][t]: batch_output_original[:,t,:] for t in range(output_seq_len)})

            _, loss_t = sess.run([rnn_model['train_op'], rnn_model['loss']], feed_dict)

            final_preds = sess.run(rnn_model['reshaped_outputs'], feed_dict)

        

            if i%5 == 0:
                print("loss after %d/%d iteractions : %.3f" %(i, total_iteractions, loss_t))

                temp_saver = rnn_model['saver']()
                name = '%d pre_days, %d gap, %s loss_function, multivariate_%d_iteractions' %(pre_days, gap, loss_function, i)
                saved_iteractions.append(name)
                path_to_city = './model_files/model/%s' %(city)
                save_path = temp_saver.save(sess, os.path.join(path_to_city, name))
                # print("Checkpoint saved at: ", save_path)

            losses.append(loss_t)
            
    
    # step 3 : validate on the dev set, select the best model and predict on the aggr data and test data

    # 3.1 predict_one_day = True,
    if predict_one_day :


        # predicting using different model on dev set
        rnn_model = build_graph(feed_previous=True, 
                                input_seq_len=input_seq_len, 
                                output_seq_len=output_seq_len, 
                                hidden_dim=hidden_dim, 
                                input_dim=input_dim, 
                                output_dim=output_dim, 
                                num_stacked_layers=num_stacked_layers, 
                                learning_rate=learning_rate,
                                lambda_l2_reg=lambda_l2_reg,
                                GRADIENT_CLIPPING=GRADIENT_CLIPPING,
                                loss_function=loss_function,
                                use_norm_data=use_norm_data)

        aver_smapes_best = 10
        model_preds_on_dev = None

        for name in saved_iteractions :
            
            # predict day one 
            init = tf.global_variables_initializer()
            with tf.Session() as sess:

                sess.run(init)
                
                # print("Using checkpoint: ", name)

                saver = rnn_model['saver']().restore(sess,  os.path.join(path_to_city, name))       
                feed_dict = {rnn_model['enc_inp'][t]: dev_x[:, t, :] for t in range(input_seq_len)} # batch prediction
                feed_dict.update({rnn_model['target_seq'][t]: np.zeros([dev_x.shape[0], output_dim], dtype=np.float32) for t in range(output_seq_len)})
                
                y_day_one = sess.run(rnn_model['reshaped_outputs'], feed_dict)
                y_day_one = [np.expand_dims(pred, 1) for pred in y_day_one]
                y_day_one = np.concatenate(y_day_one, axis = 1)        # y_day_one shape (m, 24, 105)

                # convert y_day_one to the same shape as x
                x_day_one = pad(y_day_one, dev_x_day_one, X_feature_filters, y_feature_filters)  # (m,24,385)
                dev_x_shift = np.concatenate([dev_x[:,24:,:], x_day_one], axis=1)
                print(dev_x_shift.shape)


            # predict day two
            init = tf.global_variables_initializer()
            with tf.Session() as sess:
                sess.run(init)
                saver = rnn_model['saver']().restore(sess,  os.path.join(path_to_city, name))          
                feed_dict = {rnn_model['enc_inp'][t]: dev_x_shift[:, t, :] for t in range(input_seq_len)} # batch prediction
                feed_dict.update({rnn_model['target_seq'][t]: np.zeros([dev_x_shift.shape[0], output_dim], dtype=np.float32) for t in range(output_seq_len)})
                
                y_day_two = sess.run(rnn_model['reshaped_outputs'], feed_dict)
                y_day_two = [np.expand_dims(pred, 1) for pred in y_day_two]
                y_day_two = np.concatenate(y_day_two, axis = 1)        # y_day_one shape (m, 24, 105)

            final_preds = np.concatenate([y_day_one, y_day_two], axis=1)  # (m, 48, 105)

            if use_norm_data : 
                aver_smapes, smapes_of_features, model_preds_on_dev = SMAPE_on_dataset_v1(dev_y, final_preds, output_features, statistics, 1)
            else :
                aver_smapes, smapes_of_features = SMAPE_on_dataset_v1(dev_y, final_preds, output_features, None, 1)
                model_preds_on_dev = final_preds
                

            print("name : %s, smape : %s" %(name, aver_smapes))
            
            if aver_smapes < aver_smapes_best :
                aver_smapes_best = aver_smapes
                model_preds_on_dev = model_preds_on_dev  
                model_name = name
              
        print("best iters:", model_name)



        # aqqr
        init = tf.global_variables_initializer()
        
        # day one
        with tf.Session() as sess:
            sess.run(init)

            saver = rnn_model['saver']().restore(sess,  os.path.join(path_to_city, model_name))

            feed_dict = {rnn_model['enc_inp'][t]: aggr_x[:, t, :] for t in range(input_seq_len)} # batch prediction
            feed_dict.update({rnn_model['target_seq'][t]: np.zeros([aggr_x.shape[0], output_dim], dtype=np.float32) for t in range(output_seq_len)})
            y_day_one = sess.run(rnn_model['reshaped_outputs'], feed_dict)
        
            y_day_one = [np.expand_dims(pred, 1) for pred in y_day_one]
            y_day_one = np.concatenate(y_day_one, axis = 1)


        # convert y_day_one to the same shape as x
        x_day_one = pad(y_day_one, aggr_x_day_one, X_feature_filters, y_feature_filters)  # (m,24,385)
        aggr_x_shift = np.concatenate([aggr_x[:,24:,:], x_day_one], axis=1)
        print(aggr_x_shift.shape)


        # predict day two
        init = tf.global_variables_initializer()
        with tf.Session() as sess:
            sess.run(init)
            saver = rnn_model['saver']().restore(sess,  os.path.join(path_to_city, name))          
            feed_dict = {rnn_model['enc_inp'][t]: aggr_x_shift[:, t, :] for t in range(input_seq_len)} # batch prediction
            feed_dict.update({rnn_model['target_seq'][t]: np.zeros([aggr_x_shift.shape[0], output_dim], dtype=np.float32) for t in range(output_seq_len)})
            
            y_day_two = sess.run(rnn_model['reshaped_outputs'], feed_dict)
            y_day_two = [np.expand_dims(pred, 1) for pred in y_day_two]
            y_day_two = np.concatenate(y_day_two, axis = 1)        # y_day_one shape (m, 24, 105)

        final_aggr_preds = np.concatenate([y_day_one, y_day_two], axis=1)  # (m, 48, 105)
            
        if use_norm_data :
            _, _, model_preds_on_aggr = SMAPE_on_dataset_v1(final_aggr_preds, final_aggr_preds, output_features, statistics, 1) # 仅仅是为了计算预测值
        else :
            model_preds_on_aggr = final_aggr_preds

        print("final_aggr_preds shape :", final_aggr_preds.shape)


        # 加载最好的模型
        init = tf.global_variables_initializer()

        with tf.Session() as sess:
            sess.run(init)

            saver = rnn_model['saver']().restore(sess,  os.path.join(path_to_city, model_name))

            feed_dict = {rnn_model['enc_inp'][t]: X_predict[:, t, :] for t in range(input_seq_len)} # batch prediction
            feed_dict.update({rnn_model['target_seq'][t]: np.zeros([X_predict.shape[0], output_dim], dtype=np.float32) for t in range(output_seq_len)})

            y_test_preds_day_one = sess.run(rnn_model['reshaped_outputs'], feed_dict)
            y_test_preds_day_one = [np.expand_dims(pred, 1) for pred in y_test_preds_day_one]
            y_test_preds_day_one = np.concatenate(y_test_preds_day_one, axis = 1)


        # weature forecast 
        meo_forecast = generate_meo_forecast(city=city, 
                                              station_list=station_list, 
                                              X_meo_list=X_meo_list, 
                                              use_norm_data=use_norm_data)

        # convert y_day_one to the same shape as x
        x_day_one = concate_aq_meo(y_test_preds_day_one, meo_forecast, X_feature_filters, X_aq_list, X_meo_list)
        test_x_shift = np.concatenate([X_predict[:,24:,:], x_day_one], axis=1)
        print(test_x_shift.shape)

        # predict day two
        init = tf.global_variables_initializer()
        with tf.Session() as sess:
            sess.run(init)
            saver = rnn_model['saver']().restore(sess,  os.path.join(path_to_city, name))          
            feed_dict = {rnn_model['enc_inp'][t]: test_x_shift[:, t, :] for t in range(input_seq_len)} # batch prediction
            feed_dict.update({rnn_model['target_seq'][t]: np.zeros([test_x_shift.shape[0], output_dim], dtype=np.float32) for t in range(output_seq_len)})
            
            y_test_preds_day_two = sess.run(rnn_model['reshaped_outputs'], feed_dict)
            y_test_preds_day_two = [np.expand_dims(pred, 1) for pred in y_test_preds_day_two]
            y_test_preds_day_two = np.concatenate(y_test_preds_day_two, axis = 1)        # y_day_one shape (m, 24, 105)

        final_test_preds = np.concatenate([y_test_preds_day_one, y_test_preds_day_two], axis=1)  # (m, 48, 105)

        if use_norm_data :
                _, _, model_preds_on_test = SMAPE_on_dataset_v1(final_test_preds, final_test_preds, output_features, statistics, 1) # 仅仅是为了计算预测值
        else :
             model_preds_on_test = final_test_preds


        return aver_smapes_best, model_preds_on_dev, dev_y_original, model_preds_on_aggr, aggr_y_original, model_preds_on_test, output_features  # 将在这种情况下表现最好的模型 的预测结果 和 模型的位置信息返回



    # 3.2 predict_one_day = False
    else :


        # predicting using different model on dev set
        rnn_model = build_graph(feed_previous=True, 
                                input_seq_len=input_seq_len, 
                                output_seq_len=output_seq_len, 
                                hidden_dim=hidden_dim, 
                                input_dim=input_dim, 
                                output_dim=output_dim, 
                                num_stacked_layers=num_stacked_layers, 
                                learning_rate=learning_rate,
                                lambda_l2_reg=lambda_l2_reg,
                                GRADIENT_CLIPPING=GRADIENT_CLIPPING,
                                loss_function=loss_function,
                                use_norm_data=use_norm_data)

        aver_smapes_best = 10
        model_preds_on_dev = None

        for name in saved_iteractions :

            init = tf.global_variables_initializer()
            with tf.Session() as sess:

                sess.run(init)
                
                # print("Using checkpoint: ", name)

                saver = rnn_model['saver']().restore(sess,  os.path.join(path_to_city, name))
                
                feed_dict = {rnn_model['enc_inp'][t]: dev_x[:, t, :] for t in range(input_seq_len)} # batch prediction
                feed_dict.update({rnn_model['target_seq'][t]: np.zeros([dev_x.shape[0], output_dim], dtype=np.float32) for t in range(output_seq_len)})
                
                final_preds = sess.run(rnn_model['reshaped_outputs'], feed_dict)


                final_preds = [np.expand_dims(pred, 1) for pred in final_preds]
                final_preds = np.concatenate(final_preds, axis = 1)

                if use_day_model :
                    if generate_mean and generate_range :
                        mean_preds = final_preds[:, :, :int(output_dim/2)]     # (m, output_seq_len, output_dim/2)
                        mean_preds = expand_final_preds_dim(mean_preds)   # (m, 48, output_dim/2)

                        range_preds = final_preds[:, :, int(output_dim/2):]    # (m, output_seq_len, output_dim/2) predictions of ranges
                        m = final_preds.shape[0]
                        unscaled_day_trend = expand_day_trend(m, day_trend)  # (m, 48, features)
                        scaled_day_trend = scale_day_trend(range_preds, unscaled_day_trend)
                        
                        final_preds = mean_preds + scaled_day_trend
                    else : 
                        final_preds = expand_final_preds_dim(final_preds)
                        # print("is final_preds NAN ?", np.isnan(final_preds).any().any())
                        m = final_preds.shape[0]
                        final_preds += expand_day_trend(m, day_trend)

                if use_norm_data : 
                    aver_smapes, smapes_of_features, model_preds_on_dev = SMAPE_on_dataset_v1(dev_y, final_preds, output_features, statistics, 1)
                else :
                    aver_smapes, smapes_of_features = SMAPE_on_dataset_v1(dev_y, final_preds, output_features, None, 1)
                    model_preds_on_dev = final_preds
                    

                print("name : %s, smape : %s" %(name, aver_smapes))
                
                if aver_smapes < aver_smapes_best :
                    aver_smapes_best = aver_smapes
                    model_preds_on_dev = model_preds_on_dev  
                    model_name = name
              
        print("best iters:", model_name)

        # 加载最好的模型
        init = tf.global_variables_initializer()

        with tf.Session() as sess:
            sess.run(init)

            saver = rnn_model['saver']().restore(sess,  os.path.join(path_to_city, model_name))

            feed_dict = {rnn_model['enc_inp'][t]: X_predict[:, t, :] for t in range(input_seq_len)} # batch prediction
            feed_dict.update({rnn_model['target_seq'][t]: np.zeros([X_predict.shape[0], output_dim], dtype=np.float32) for t in range(output_seq_len)})

            final_test_preds = sess.run(rnn_model['reshaped_outputs'], feed_dict)
        
            final_test_preds = [np.expand_dims(pred, 1) for pred in final_test_preds]
            final_test_preds = np.concatenate(final_test_preds, axis = 1)

            if use_day_model :
                if generate_mean and generate_range :
                    mean_preds = final_test_preds[:, :, :int(output_dim/2)]     # (m, output_seq_len, output_dim/2)
                    mean_preds = expand_final_preds_dim(mean_preds)   # (m, 48, output_dim/2)

                    range_preds = final_test_preds[:, :, int(output_dim/2):]    # (m, output_seq_len, output_dim/2) predictions of ranges
                    m = final_test_preds.shape[0]
                    unscaled_day_trend = expand_day_trend(m, day_trend)  # (m, 48, features)
                    scaled_day_trend = scale_day_trend(range_preds, unscaled_day_trend)
                    
                    final_test_preds = mean_preds + scaled_day_trend
                else : 
                    final_test_preds = expand_final_preds_dim(final_test_preds)
                    # print("is final_preds NAN ?", np.isnan(final_preds).any().any())
                    m = final_test_preds.shape[0]
                    final_test_preds += expand_day_trend(m, day_trend)
            
            if use_norm_data :
                _, _, model_preds_on_test = SMAPE_on_dataset_v1(final_test_preds, final_test_preds, output_features, statistics, 1) # 仅仅是为了计算预测值
            else :
                 model_preds_on_test = final_test_preds


        # aqqr
        init = tf.global_variables_initializer()

        with tf.Session() as sess:
            sess.run(init)

            saver = rnn_model['saver']().restore(sess,  os.path.join(path_to_city, model_name))

            feed_dict = {rnn_model['enc_inp'][t]: aggr_x[:, t, :] for t in range(input_seq_len)} # batch prediction
            feed_dict.update({rnn_model['target_seq'][t]: np.zeros([aggr_x.shape[0], output_dim], dtype=np.float32) for t in range(output_seq_len)})
            final_aggr_preds = sess.run(rnn_model['reshaped_outputs'], feed_dict)
        
            final_aggr_preds = [np.expand_dims(pred, 1) for pred in final_aggr_preds]
            final_aggr_preds = np.concatenate(final_aggr_preds, axis = 1)

            if use_day_model :
                if generate_mean and generate_range :
                    mean_preds = final_aggr_preds[:, :, :int(output_dim/2)]     # (m, output_seq_len, output_dim/2)
                    mean_preds = expand_final_preds_dim(mean_preds)   # (m, 48, output_dim/2)

                    range_preds = final_aggr_preds[:, :, int(output_dim/2):]    # (m, output_seq_len, output_dim/2) predictions of ranges
                    m = final_aggr_preds.shape[0]
                    unscaled_day_trend = expand_day_trend(m, day_trend)  # (m, 48, features)
                    scaled_day_trend = scale_day_trend(range_preds, unscaled_day_trend)
                    
                    final_aggr_preds = mean_preds + scaled_day_trend
                else : 
                    final_aggr_preds = expand_final_preds_dim(final_aggr_preds)
                    # print("is final_preds NAN ?", np.isnan(final_preds).any().any())
                    m = final_aggr_preds.shape[0]
                    final_aggr_preds += expand_day_trend(m, day_trend)
            
            if use_norm_data :
                _, _, model_preds_on_aggr = SMAPE_on_dataset_v1(final_aggr_preds, final_aggr_preds, output_features, statistics, 1) # 仅仅是为了计算预测值
            else :
                model_preds_on_aggr = final_aggr_preds
           
        return aver_smapes_best, model_preds_on_dev, dev_y_original, model_preds_on_aggr, aggr_y_original, model_preds_on_test, output_features  # 将在这种情况下表现最好的模型 的预测结果 和 模型的位置信息返回
