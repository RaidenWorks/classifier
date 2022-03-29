# Method to plot histogram and boxplot, and print data
import matplotlib.pyplot as plt


def histo_boxplot(dataframe, column, bin_number, width=8, height=8):
    """Plots a histogram and boxplot of dataframe[column]

    :param dataframe: dataframe
    :param column: column of dataframe to plot
    :param bin_number: number of bins for histogram
    :param width: width of plot
    :param height: height of plot
    :return: histogram and boxplot of dataframe[column]

    >>> histo_boxplot(df, 'column1', 10)
    None
    """
    # Get variable to examine
    var = dataframe[column]

    # Create a figure
    fig, ax = plt.subplots(2, 1, figsize=(width, height))  # distribution and boxplot
    # fig = plt.figure(figsize = (width, height))  # single distribution plot

    # Plot a histogram of specific column data
    ax[0].hist(var, bins=bin_number)
    ax[0].set_ylabel('Frequency')
    ax[0].set_title(column + ' Distribution')

    # Get statistics
    value_mean = var.mean()
    value_med = var.median()
    value_mod = var.mode()[0]
    value_min = var.min()
    value_max = var.max()
    value_std = var.std()

    # Plot lines for statistics
    ax[0].axvline(x=value_mean, color='cyan', linestyle='dashed', linewidth=2)
    ax[0].axvline(x=value_med, color='red', linestyle='dashed', linewidth=2)
    ax[0].axvline(x=value_mod, color='yellow', linestyle='dashed', linewidth=2)
    ax[0].axvline(x=value_min, color='grey', linestyle='dashed', linewidth=2)
    ax[0].axvline(x=value_max, color='grey', linestyle='dashed', linewidth=2)

    # Plot the boxplot
    ax[1].boxplot(var, vert=False)
    ax[1].set_xlabel('Value')

    # Print
    print('Minimum: {:.2f}\nMean(cyan): {:.2f}\nMedian(red): {:.2f}'
          .format(value_min, value_mean, value_med))
    print('Mode(yellow): {:.2f}\nMaximum: {:.2f}\nStdDev: {:.2f}'
          .format(value_min, value_mean, value_med))
