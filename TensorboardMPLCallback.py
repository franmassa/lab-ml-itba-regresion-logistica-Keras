import tensorflow as tf
from tensorflow.keras.callbacks import Callback
import numpy as np
from IPython.display import clear_output
from matplotlib import pyplot as plt
from matplotlib.gridspec import GridSpec
from sklearn.metrics import roc_auc_score, roc_curve
from scipy.spatial import Delaunay

import io

from tensorboard.plugins.mesh import summary_v2 as mesh_summary

def plot_to_image(figure):
    """Converts the matplotlib plot specified by 'figure' to a PNG image and
    returns it. The supplied figure is closed and inaccessible after this call."""
    # Save the plot to a PNG in memory.
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    # Closing the figure prevents it from being displayed directly inside
    # the notebook.
    plt.close(figure)
    buf.seek(0)
    # Convert PNG buffer to TF image
    image = tf.image.decode_png(buf.getvalue(), channels=4)
    # Add the batch dimension
    image = tf.expand_dims(image, 0)
    return image

def get_loss_by_sample(y_true, y_pred, eps=1e-15):
    # get loss of each observation
    losses = - y_true * np.log(y_pred + eps) - (1-y_true) * np.log(1-y_pred + eps)
    idxs = np.where(np.isnan(losses))[0]
    if len(idxs)>0:
        print(y_pred[idxs[0]])
    return losses

def plotBoundary(model, X, y, X_transformed, X_grid, X_grid_transformed, class_0, class_1, acc, loss, bins=100, file_writer=None, epoch=0, n_colors = 100):
    eps=1e-8
    clear_output(wait=True)
    fig=plt.figure(figsize=(20,11))
    gs=GridSpec(3, 3) # 2 rows, 3 columns    
    
    axLoss=fig.add_subplot(gs[0,2]) # First row, first column
    axAcc=fig.add_subplot(gs[1,2]) # First row, second column
    axLossHist = fig.add_subplot(gs[2,1])
    axLogOddsHist = fig.add_subplot(gs[2,0])
    
    predictions = model.predict_proba(X_transformed).reshape(-1)
    predictions_0 = predictions[class_0]
    predictions_1 = predictions[class_1]
    log_odds_0 = np.log(eps + predictions_0/(1 - predictions_0 + eps))
    log_odds_1 = np.log(eps + predictions_1/(1 - predictions_1 + eps))
    axLogOddsHist.hist(log_odds_0, bins, color='r')
    axLogOddsHist.hist(log_odds_1, bins, color='b', alpha=0.5)

    losses_0 = get_loss_by_sample(np.zeros(len(predictions_0)), predictions_0)
    losses_1 = get_loss_by_sample(np.ones(len(predictions_1)), predictions_1)

    axLossHist.hist(losses_0, bins, color='r')
    axLossHist.hist(losses_1, bins, color='b', alpha=0.5)

    axROC = fig.add_subplot(gs[2,2])
    auROC = roc_auc_score(y, predictions)
    axROC.set_title(f'ROC curve - AuROC:{auROC:.4f}')
    fpr, tpr, thres = roc_curve(y, predictions)
    axROC.plot(fpr, tpr)

    
    Z = model.predict_proba(X_grid_transformed).reshape(-1)
    Z = Z.reshape(X_grid[0].shape)
    cm = plt.cm.RdBu
    
    plot_contour = False
    if plot_contour:
        # Plot contour threshold
        ax=fig.add_subplot(gs[:2,:2]) # Second row, span all columns
        ax.contour(X_grid[0], X_grid[1], Z, (0.5,), colors='k', linewidths=0.5)

        # Plot contour surface
        cf = ax.contourf(X_grid[0], X_grid[1], Z, n_colors, vmin=0., vmax=1., cmap=cm, alpha=.8)
        plt.colorbar(cf, ax=ax)

        # Plot Points
        ax.scatter(X[class_1][:,0], X[class_1][:,1], color='b', s=5, alpha=0.5)
        ax.scatter(X[class_0][:,0], X[class_0][:,1], color='r', s=5, alpha=0.5)
    else:
        # Plot Surface
        ax=fig.add_subplot(gs[:2,:2], projection='3d') # Second row, span all columns
        ax.plot_surface(X_grid[0], X_grid[1], Z, cmap=cm, linewidth=0, antialiased=False, alpha=0.5, vmin=0., vmax=1.)
        ax.scatter(X[:,0], X[:,1], y, marker='o', c=y ,cmap=cm, vmin=0., vmax=1., alpha=1.0)
        # ax.plot_surface(X_grid[0], X_grid[1], 0.5*np.ones(Z.shape), alpha= 0.5, cmap='gray')
        ax.contour(X_grid[0], X_grid[1], Z, (0.5,), colors='k', linewidths=2)
    
    axAcc.plot(acc)

    if len(acc)==0 or acc[0] is None:
        loss, acc = model.evaluate(X_transformed, y, verbose=0)
        axAcc.set_title(f'Accuracy: {acc:.4f}')
        axLoss.set_title(f'Cross Entropy: {loss:.4f}')
    else:
        axAcc.set_title(f'Accuracy: {acc[-1]:.4f}')
        axLoss.set_title(f'Cross Entropy: {loss[-1]:.4f}')

    axLoss.plot(loss)

    axLossHist.set_title('Cross Entropy Histogram')
    axLogOddsHist.set_title('Log odds Histogram')
    
    if file_writer is not None:
        print('image to tensorboard')
        
        with file_writer.as_default():
            image = plot_to_image(fig)
            tf.summary.image("Training data", image, step=epoch+1)
            
            
            config_dict = {
                'camera': {'cls': 'PerspectiveCamera', 'fov': 75},
                'lights': [
                    {
                      'cls': 'AmbientLight', 
                      'color': '#ffffff',
                      'intensity': 0.75,
                    }, {
                      'cls': 'DirectionalLight',
                      'color': '#ffffff',
                      'intensity': 0.75,
                      'position': [0, -1, 2],
                    }],
                'material': {
                  'cls': 'MeshStandardMaterial',
                  #'roughness': 1,
                  'opacity': 0.8,
                  'transparent': True,
                  #'metalness': 0
                }
            }
            cmap = plt.cm.get_cmap('RdBu')
            colors = np.array([[list(cmap(p)[:3]) for p in Z.reshape(-1)]])*255
            # colors = np.array([[0.5/255] + list(cmap(p)[:3]) for p in Z.reshape(-1)])*255
            print(colors[:3])
            tri = Delaunay( np.array([X_grid[0].reshape(-1), X_grid[1].reshape(-1)]).T )
            faces = np.array([tri.simplices.copy()])
            mesh = np.array([np.array([X_grid[0].reshape(-1), X_grid[1].reshape(-1), Z.reshape(-1)]).T])
            
            
            mesh_summary.mesh('mesh/surface', 
                              vertices=mesh,
                              faces=faces,
                              colors=colors, 
                              config_dict=config_dict,
                              step=epoch+1)
            
# Grafico de puntos rojos (No tiene mucho sentido si no se puede superponer)
#             mesh_summary.mesh('mesh/points', 
#                               vertices=np.array([[X[class_0][:,0], X[class_0][:,1], y[class_0]]]), 
#                               colors=np.array([[[255, 0, 0]]*len(y[class_0])]),
#                               step=epoch+1)
            
# TensorboardX
#             summaryWriter.add_mesh('mesh/surface', 
#                               vertices=mesh, 
#                               faces=faces,
#                               colors=colors, 
#                               config_dict=config_dict,
#                               global_step=epoch+1)
            
#             summaryWriter.add_mesh('mesh/surface', 
#                               vertices=np.array([[X[class_0][:,0], X[class_0][:,1], y[class_0]]]), 
#                               colors=np.array([[[255, 0, 0]]*len(y[class_0])]),
#                               config_dict=config_dict,
#                               global_step=epoch+1)
            
            
            
    else:
        plt.show()


class PlotCallbackTB(Callback):     
    def __init__(self, data, labels, plots_every_batches=100, N = 300, bins=100, degree=1, feat_eng_transform=None, logdir=None):
        # feat_eng_transform transform example
        #   polyFeat = PolynomialFeatures(degree=degree, interaction_only=False, include_bias=False)
        #   polyFeat.fit_transform(data)
        #   def feat_eng_transform(data):
        #       return polyFeat.transform(data)
        tf.summary.experimental.set_step(1)
        
        if logdir is not None:
            self.file_writer = tf.summary.create_file_writer(logdir)
#             self.summaryWriter = SummaryWriter(logdir+'X')
        else:
            self.file_writer = None
#             self.summaryWriter = None
            
        if feat_eng_transform is None:
            feat_eng_transform = lambda data: data
            
        self.data_transformed = feat_eng_transform(data)
        
        self.plots_every_batches = plots_every_batches
        self.bins = bins
        self.N = N
        self.data = data
        self.labels = labels
        mins = data[:,:2].min(axis=0)
        maxs = data[:,:2].max(axis=0)
        X_lin = np.linspace(mins[0], maxs[0], self.N)
        Y_lin = np.linspace(mins[1], maxs[1], self.N)
        self.X, self.Y = np.meshgrid(X_lin, Y_lin)
        self.Z_shape = self.X.shape
        grid_data = np.c_[self.X.flatten(), self.Y.flatten()]
        self.grid_data_transformed = feat_eng_transform(grid_data)
        self.acc = []
        self.loss = []
        self.class_1 = labels == 1
        self.class_0 = labels == 0
        
    def on_train_begin(self, logs={}):
        plotBoundary(self.model, self.data, self.labels, self.data_transformed, 
                     (self.X, self.Y), self.grid_data_transformed, self.class_0, 
                     self.class_1, self.acc, self.loss, self.bins, 
                     file_writer=self.file_writer, epoch=0)
        return
    
    def on_epoch_end(self, epoch, logs={}):
        print(logs)
        self.acc.append(logs.get('accuracy'))
        self.loss.append(logs.get('loss'))
        plotBoundary(self.model, self.data, self.labels, self.data_transformed, 
                     (self.X, self.Y), self.grid_data_transformed, self.class_0, 
                     self.class_1, self.acc, self.loss, self.bins, 
                     file_writer=self.file_writer, epoch=epoch)
        return
    
    def on_batch_end(self, batch, logs={}):
        # if batch%self.plots_every_batches == 0:
        #    self.acc.append(logs.get('acc'))
        #    self.loss.append(logs.get('loss'))
        #    plotBoundary(self.model)
        return
    
    def on_train_end(self, batch, logs=None):
        if self.file_writer is not None:
            self.file_writer.flush()
            self.file_writer.close()
            print('summary writer closed!')