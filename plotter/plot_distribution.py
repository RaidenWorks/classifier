# Method to plot histogram and boxplot, and print data
import matplotlib.pyplot as plt


class Distribution:
    """Get metrics of a pandas series and plot its histogram and boxplot

    :param dataframe: dataframe
    :param column: column name of dataframe
    :param bin_number: number of bins for histogram plot
    :param width: width of plot
    :param height: height of plot

    :ivar mean: average of dataframe[column] series
    :ivar median: median of dataframe[column] series
    :ivar mode: most common value of dataframe[column] series
    :ivar min: minimum value of dataframe[column] series
    :ivar max: maximum value of dataframe[column] series
    :ivar std: standard deviation of dataframe[column] series
    :ivar histo_boxplot: plots a histogram and boxplot of dataframe[column] series
    """
    def __init__(self, dataframe, column, bin_number=10, width=8, height=8):
        self.series = dataframe[column]
        self.column = column
        self.bin_number = bin_number
        self.width = width
        self.height = height
        # pre-calculate metrics with non-public methods
        self.mean = self._mean()
        self.median = self._median()
        self.mode = self._mode()
        self.min = self._min()
        self.max = self._max()
        self.std = self._std()

    # non-public method to return mean
    def _mean(self):
        return self.series.mean()

    # non-public method to return median
    def _median(self):
        return self.series.median()

    # non-public method to return mode
    def _mode(self):
        return self.series.mode()[0]

    # non-public method to return minimum
    def _min(self):
        return self.series.min()

    # non-public method to return maximum
    def _max(self):
        return self.series.max()

    # non-public method to return standard deviation
    def _std(self):
        return self.series.std()

    # public method to plot histogram and boxplot
    def histo_boxplot(self):
        # Create a figure
        fig, ax = plt.subplots(2, 1, figsize=(self.width, self.height))  # distribution and boxplot
        # fig = plt.figure(figsize = (width, height))  # single distribution plot

        # Plot a histogram of specific column data
        ax[0].hist(self.series, bins=self.bin_number)
        ax[0].set_ylabel('Frequency')
        ax[0].set_title(self.column + ' Distribution')

        # Plot lines for statistics
        ax[0].axvline(x=self.mean, color='cyan', linestyle='dashed', linewidth=2)
        ax[0].axvline(x=self.median, color='red', linestyle='dashed', linewidth=2)
        ax[0].axvline(x=self.mode, color='yellow', linestyle='dashed', linewidth=2)
        ax[0].axvline(x=self.min, color='grey', linestyle='dashed', linewidth=2)
        ax[0].axvline(x=self.max, color='grey', linestyle='dashed', linewidth=2)

        # Plot the boxplot
        ax[1].boxplot(self.series, vert=False)
        ax[1].set_xlabel('Value')

        # Print
        print('Minimum: {:.2f}\nMean(cyan): {:.2f}\nMedian(red): {:.2f}'
              .format(self.min, self.mean, self.median))
        print('Mode(yellow): {:.2f}\nMaximum: {:.2f}\nStdDev: {:.2f}'
              .format(self.mode, self.max, self.std))


# retained this method as reference
def histo_boxplot(dataframe, column, bin_number=10, width=8, height=8):
    """Plots a histogram and boxplot of dataframe[column]

    :param dataframe: dataframe
    :param column: column of dataframe to plot
    :param bin_number: number of bins for histogram
    :param width: width of plot
    :param height: height of plot
    :return: None; a histogram and boxplot of dataframe[column] is shown

    >>> histo_boxplot(df, 'column_name', 10)
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
          .format(value_mod, value_max, value_std))
