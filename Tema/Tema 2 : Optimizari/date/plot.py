import matplotlib.pyplot as plt


def read(file_name):
    with open(file_name, "r") as file:
        file.readline()
        x_list = []
        y_list = []
        for line in file.readlines():
            l = line.strip().split(", ")
            x_list.append(int(l[0]))
            y_list.append(float(l[1]))

        return x_list, y_list


def draw_plot(x, y, x_name, y_name, title):
    plt.plot(x, y)
    plt.xlabel(x_name)
    plt.ylabel(y_name)
    plt.title(title)
    plt.show()


def draw_tow(x1, y1, x2, y2, x_name, y_name, title):
    plt.plot(x1, y1, label="gcc")
    plt.plot(x2, y2, label="icc")

    plt.xlabel(x_name)
    plt.ylabel(y_name)

    plt.title(title)
    plt.legend()
    plt.show()


def draw_4(x, y1, y2, y3, y4, title):
    plt.plot(x, y1, label="neopt")
    plt.plot(x, y2, label="opt-f")
    plt.plot(x, y3, label="opt-m")
    plt.plot(x, y4, label="blas")

    plt.xlabel("N")
    plt.ylabel("Time")

    plt.title(title)
    plt.legend()
    plt.show()


def draw_8(x, y1, y2, y3, y4, y5, y6, y7, y8):
    plt.plot(x, y1, label="neopt-gcc", color="cyan")
    plt.plot(x, y2, label="opt-f-gcc", color="lime")
    plt.plot(x, y3, label="opt-m-gcc", color="gold")
    plt.plot(x, y4, label="blas-gcc", color="navy")

    plt.plot(x, y5, label="neopt-icc", color="darkblue")
    plt.plot(x, y6, label="opt-f-icc", color="red")
    plt.plot(x, y7, label="opt-m-icc", color="coral")
    plt.plot(x, y8, label="blas-icc", color="yellow")

    plt.xlabel("N")
    plt.ylabel("Time")

    plt.title("GCC vs ICC")
    plt.legend()
    plt.show()


x_blas_gcc, y__blas_gcc = read("tema2_blas_gcc.txt")
x_neopt_gcc, y__neopt_gcc = read("tema2_neopt_gcc.txt")
x_opt_m_gcc, y__opt_m_gcc = read("tema2_opt_m_gcc.txt")
x_opt_f_gcc, y__opt_f_gcc = read("tema2_opt_f_gcc.txt")

x_blas_icc, y__blas_icc = read("tema2_blas_icc.txt")
x_neopt_icc, y__neopt_icc = read("tema2_neopt_icc.txt")
x_opt_m_icc, y__opt_m_icc = read("tema2_opt_m_icc.txt")
x_opt_f_icc, y__opt_f_icc = read("tema2_opt_f_icc.txt")
draw_plot(x_neopt_gcc, y__neopt_gcc, "N", "Time", "Neoptimizat - GCC")
draw_plot(x_opt_f_gcc, y__opt_f_gcc, "N", "Time", "Compiler Flag - GCC")
draw_plot(x_opt_m_gcc, y__opt_m_gcc, "N", "Time", "Optimizat - GCC")
draw_plot(x_blas_gcc, y__blas_gcc, "N", "Time", "BLAS -GCC")

draw_plot(x_neopt_icc, y__neopt_icc, "N", "Time", "Neoptimizat - ICC")
draw_plot(x_opt_f_icc, y__opt_f_icc, "N", "Time", "Compiler Flag - ICC")
draw_plot(x_opt_m_icc, y__opt_m_icc, "N", "Time", "Optimizat - ICC")
draw_plot(x_blas_icc, y__blas_icc, "N", "Time", "BLAS -ICC")
draw_tow(x_neopt_gcc, y__neopt_gcc, x_neopt_icc, y__neopt_icc, "N", "Time", "Neoptimizat GCC vs ICC")
draw_tow(x_opt_m_gcc, y__opt_m_gcc, x_opt_m_icc, y__opt_m_icc, "N", "Time", "Optimizat GCC vs ICC")
draw_tow(x_opt_f_gcc, y__opt_f_gcc, x_opt_f_icc, y__opt_f_icc, "N", "Time", "Compiler Flag GCC vs ICC")
draw_tow(x_blas_gcc, y__blas_gcc, x_blas_icc, y__blas_icc, "N", "Time", "BLAS GCC vs ICC")
draw_4(x_blas_gcc, y__neopt_gcc, y__opt_f_gcc, y__opt_m_gcc, y__blas_gcc, "GCC")

draw_4(x_blas_icc, y__neopt_icc, y__opt_f_icc, y__opt_m_icc, y__blas_icc, "ICC")


draw_8(x_blas_gcc, y__neopt_gcc, y__opt_f_gcc, y__opt_m_gcc, y__blas_gcc, y__neopt_icc, y__opt_f_icc, y__opt_m_icc,
       y__blas_icc)
