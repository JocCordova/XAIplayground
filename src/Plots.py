from DataExploration import Plotter

plotter = Plotter()


def plot_columns():
    plotter.plot_column("Gender")
    plotter.plot_column("Nationality")
    plotter.plot_column("StageID")
    plotter.plot_column("Class")


def plot_columns_grouped():
    plotter.plot_column_grouped("Class", "StudentAbsenceDays")
    plotter.plot_column_grouped("Class", "Gender")
    plotter.plot_column_grouped("Class", "Topic")
    plotter.plot_column_grouped("Class", "Semester")
    plotter.plot_column_grouped("Class", "Relation")


def hist_columns_grouped():
    plotter.hist_plot_column_grouped("RaisedHands")
    plotter.hist_plot_column_grouped("VisitedResources")
    plotter.hist_plot_column_grouped("AnnouncementsViewed")
    plotter.hist_plot_column_grouped("Discussion")


plot_columns()
plot_columns_grouped()
hist_columns_grouped()
