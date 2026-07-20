import matplotlib.pyplot as plt
import seaborn as sns


class Visualizer :

    @staticmethod
    def plot_training_history(history):
        plt.figure(figsize=(12,4))
        plt.subplot(1,2,1)
        plt.plot(history.history['accuracy'], label='Train Accuracy')
        plt.plot(history.history['val_accuracy'],label='Validation Accuracy')
        plt.title("Model Accuracy")
        plt.ylabel("Accuracy")
        plt.xlabel("Epoch")
        plt.legend()

        plt.subplot(1,2,2)
        plt.plot(history.history['loss'], label='Train Loss')
        plt.plot(history.history['val_loss'],labek='Validation Loss')
        plt.title("Model Loss")
        plt.ylabel("Loss")
        plt.xlabel("Epoch")
        plt.legend()
        
        plt.show()