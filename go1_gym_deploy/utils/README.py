from cmx import doc
from matplotlib import pyplot as plt

doc @ """ # Compute Usage on Orin with jtop """

with doc @ """ ### With ZED Camera Ultra Mode, SVGA """:
    from ml_logger import logger
    import pandas as pd

    prefix = "/alanyu/scratch/2024/01-03/154316"
    with logger.Prefix(prefix):
        gpu, ram = logger.read_metrics('GPU', "RAM")
        gpu = gpu.apply(lambda x: x[0])
        ram = ram.apply(lambda x: x[0])

    combined_df = gpu.to_frame('GPU').join(ram.to_frame('RAM'))

    print(combined_df.head())

    plt.figure(figsize=(12, 8))
    plt.plot(combined_df['GPU'], label='GPU Usage', color='blue', marker='o', linestyle='-') 
    plt.plot(combined_df['RAM'], label='RAM Usage', color='red', marker='x', linestyle='-')  

    # Adding titles and labels
    plt.title('GPU and RAM Usage Over Time')
    plt.xlabel('Time')
    plt.ylabel('Usage (%)')
    plt.legend()

    # Show the plot
    plt.grid(True)
    plt.show()
