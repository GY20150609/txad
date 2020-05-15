import tensorflow as tf
import shutil

class SparseModel:

    def __init__(self, config, params):
        self.config = config
        self.model = None
        global fm_params
        fm_params = params

    def input_fn(self, filenames, batch_size=32, num_epochs=1, perform_shuffle=False):
        return None

    def net(self,features,labels,params):
        return {}

    def model_fn(self,features, labels, mode,params):

        
        if mode == tf.estimator.ModeKeys.TRAIN:
            params["train_phase"] = True
            catches = self.net(features,labels,params)
        else:
            params["train_phase"] = False
            catches = self.net(features,labels,params)
            

        pred = catches["pred"]
        loss = catches["loss"]
        train_op = catches["train_op"]

        predictions = {"prob": pred}
        # 训练
        if mode == tf.estimator.ModeKeys.TRAIN:
            return tf.estimator.EstimatorSpec(
                mode=mode,
                predictions=predictions,
                loss=loss,
                train_op=train_op)

        # 评估
        eval_metric_ops = {
            "auc": tf.compat.v1.metrics.auc(labels, pred)
        }
        if mode == tf.estimator.ModeKeys.EVAL:
            return tf.estimator.EstimatorSpec(
                mode=mode,
                predictions=predictions,
                loss=loss,
                eval_metric_ops=eval_metric_ops)

        # 预测
        export_outputs = {
            tf.saved_model.signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY: tf.estimator.export.PredictOutput(
                predictions)}
        if mode == tf.estimator.ModeKeys.PREDICT:
            return tf.estimator.EstimatorSpec(
                mode=mode,
                predictions=predictions,
                export_outputs=export_outputs)
        return tf.estimator.EstimatorSpec(mode, loss=loss, train_op=train_op)

    def compile(self,clear_existing_model=True):
        run_config = self.config
        model_dir = './model/checkpoint/{}'.format(self.__class__.__name__)
        if clear_existing_model:
            try:
                shutil.rmtree(model_dir)
            except Exception as e:
                print(e, "at clear_existing_model")
        else:
            print("existing model cleaned at {}".format(model_dir))
        self.model = tf.estimator.Estimator(model_fn=self.model_fn, model_dir=model_dir, params=fm_params,
                                            config=run_config)

    def train(self, tr_files, va_files):
        self.model.train(input_fn=lambda: self.input_fn(tr_files, num_epochs=fm_params["num_epochs"],
                                                        batch_size=fm_params["batch_size"]))

    def evaluate(self, va_files):
        self.model.evaluate(input_fn=lambda: self.input_fn(va_files, num_epochs=1, batch_size=fm_params["batch_size"]))

    def train_and_evaluate(self, tr_files, va_files):
        evaluator = tf.estimator.experimental.InMemoryEvaluatorHook(
            estimator=self.model,
            input_fn=lambda: self.input_fn(va_files, num_epochs=1, batch_size=fm_params["batch_size"]),
            every_n_iter=fm_params["val_itrs"])
        self.model.train(
            input_fn=lambda: self.input_fn(tr_files, num_epochs=fm_params["num_epochs"],
                                           batch_size=fm_params["batch_size"]),
            hooks=[evaluator])

    def predict(self, te_files, isSave=False, numToSave=10):
        P_G = self.model.predict(input_fn=lambda: self.input_fn(te_files, num_epochs=1, batch_size=1),
                                 predict_keys="prob")
        if isSave:
            with open(te_files, 'r') as f1, open('sample.unitest', "w") as f2:
                for i in range(numToSave):
                    sample = f1.readline()
                    result = next(P_G)
                    pred = str(result['prob'])
                    f2.write('\t'.join([pred, sample]))