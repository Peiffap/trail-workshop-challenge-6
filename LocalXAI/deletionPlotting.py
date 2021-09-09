### IMPORTS ###

import os
import numpy as np
import matplotlib.pyplot as plt

### MAIN Code ###

# Base path
path = "/home_nfs/stassinsed/Workshop/results/"
# XAI methods for which we have a txt file with probabilities value
# Coming from Deletion XAI Evaluation method
mths = [('LayerCAM'), ('InputXGradient'), ('GuidedBackprop'), 
        ('Deconvolution'), ('Saliency'), ('GradCAM'),
        ('GradCAM++'),('EigenGradCAM') ,('GuidedGradCam')]



# Cluster of images given after choosing manually over the embedding (tSNE, MDE, ...)

# cluster = ['person102_bacteria_487_list',
#  'person152_bacteria_720_list',
#  'person152_bacteria_721_list',
#  'person136_bacteria_649_list',
#  'person82_bacteria_404_list',
#  'NORMAL2-IM-0316-0001_list',
#  'NORMAL2-IM-0107-0001_list',
#  'person85_bacteria_419_list',
#  'person75_virus_136_list',
#  'IM-0073-0001_list',
#  'person118_bacteria_559_list',
#  'person8_virus_27_list',
#  'person1_virus_11_list',
#  'person126_bacteria_600_list',
#  'person103_bacteria_489_list',
#  'person81_bacteria_395_list',
#  'person133_bacteria_634_list',
#  'person147_bacteria_706_list',
#  'person120_bacteria_570_list',
#  'person88_bacteria_437_list',
#  'person78_bacteria_382_list',
#  'person109_bacteria_519_list',
#  'person174_bacteria_832_list',
#  'person83_bacteria_411_list',
#  'person1_virus_12_list',
#  'person83_bacteria_412_list',
#  'person1_virus_8_list',
#  'person121_bacteria_575_list',
#  'IM-0075-0001_list',
#  'person123_bacteria_587_list',
#  'person136_bacteria_652_list',
#  'person83_bacteria_410_list',
#  'person159_bacteria_747_list',
#  'person82_bacteria_402_list',
#  'person82_bacteria_405_list',
#  'person85_bacteria_424_list',
#  'person158_bacteria_742_list',
#  'person112_bacteria_539_list',
#  'person108_bacteria_506_list',
#  'person128_bacteria_606_list',
#  'person78_bacteria_380_list',
#  'person122_bacteria_583_list',
#  'person136_bacteria_648_list',
#  'person1_virus_9_list',
#  'person78_bacteria_384_list',
#  'person111_bacteria_535_list',
#  'person80_bacteria_390_list',
#  'person114_bacteria_546_list',
#  'person28_virus_62_list',
#  'person109_bacteria_517_list',
#  'person175_bacteria_833_list',
#  'person114_bacteria_544_list',
#  'person29_virus_64_list',
#  'person124_bacteria_589_list',
#  'person86_bacteria_428_list',
#  'person85_bacteria_417_list',
#  'person81_bacteria_397_list',
#  'person158_bacteria_743_list',
#  'person132_bacteria_632_list',
#  'person141_bacteria_676_list',
#  'person109_bacteria_523_list',
#  'person139_bacteria_662_list',
#  'person141_bacteria_681_list',
#  'person85_bacteria_422_list',
#  'person1627_virus_2819_list',
#  'person161_bacteria_759_list',
#  'person85_bacteria_423_list',
#  'NORMAL2-IM-0294-0001_list',
#  'person78_bacteria_386_list',
#  'person95_bacteria_463_list',
#  'person142_bacteria_684_list',
#  'IM-0081-0001_list',
#  'person146_bacteria_704_list',
#  'person135_bacteria_647_list',
#  'person112_bacteria_538_list',
#  'NORMAL2-IM-0273-0001_list',
#  'person139_bacteria_665_list',
#  'person125_bacteria_594_list',
#  'person94_bacteria_457_list',
#  'person104_bacteria_492_list',
#  'person30_virus_69_list',
#  'NORMAL2-IM-0352-0001_list',
#  'person126_bacteria_599_list',
#  'person138_bacteria_658_list',
#     'person120_bacteria_572_list',
#  'person113_bacteria_543_list',
#  'person134_bacteria_640_list',
#  'person146_bacteria_703_list',
#  'person158_bacteria_744_list',
#  'person157_bacteria_739_list',
#  'NORMAL2-IM-0279-0001_list',
#  'person78_bacteria_378_list',
#  'person100_bacteria_478_list',
#  'person121_bacteria_578_list',
#  'NORMAL2-IM-0331-0001_list',
#  'person173_bacteria_829_list',
#  'person87_bacteria_434_list',
#  'NORMAL2-IM-0348-0001_list',
#  'person147_bacteria_707_list',
#  'person133_bacteria_635_list',
#  'person122_bacteria_582_list',
#  'person96_bacteria_464_list',
#  'person1663_virus_2876_list',
#  'person130_bacteria_627_list',
#  'NORMAL2-IM-0066-0001_list',
#  'person155_bacteria_729_list',
#  'person91_bacteria_447_list',
#  'person135_bacteria_646_list',
#  'person128_bacteria_605_list',
#  'person101_bacteria_486_list',
#  'person175_bacteria_835_list',
#  'person145_bacteria_696_list',
#  'person155_bacteria_730_list',
#  'person90_bacteria_442_list',
#  'person18_virus_49_list',
#  'IM-0101-0001_list',
#  'person1_virus_6_list',
#  'NORMAL2-IM-0307-0001_list',
#  'person1635_virus_2831_list',
#  'person131_bacteria_629_list',
#  'person99_bacteria_473_list',
#  'person109_bacteria_527_list',
#  'NORMAL2-IM-0098-0001_list',
#  'person100_bacteria_480_list',
#  'person150_bacteria_715_list',
#  'person117_bacteria_553_list',
#  'person172_bacteria_827_list',
#  'person124_bacteria_591_list',
#  'person113_bacteria_541_list',
#  'person99_bacteria_474_list',
#  'person120_bacteria_571_list',
#  'person152_bacteria_722_list',
#  'person143_bacteria_688_list',
#  'person119_bacteria_566_list',
#  'person3_virus_15_list',
#  'person136_bacteria_650_list',
#  'person109_bacteria_522_list',
#  'person1682_virus_2899_list',
#  'person109_bacteria_513_list',
#  'person91_bacteria_446_list',
#  'IM-0109-0001_list',
#  'person100_bacteria_479_list',
#  'person130_bacteria_625_list',
#  'person159_bacteria_746_list',
#  'person146_bacteria_700_list',
#  'person152_bacteria_723_list',
#  'person101_bacteria_484_list',
#  'person1685_virus_2903_list',
#  'person175_bacteria_834_list',
#  'person80_bacteria_393_list',
#  'person110_bacteria_531_list',
#  'person93_bacteria_454_list',
#  'person121_bacteria_579_list',
#  'NORMAL2-IM-0354-0001_list',
#  'person92_bacteria_450_list',
#  'person139_bacteria_666_list',
#  'NORMAL2-IM-0322-0001_list',
#  'person108_bacteria_507_list',
#  'person139_bacteria_663_list',
#  'person100_bacteria_475_list',
#  'NORMAL2-IM-0123-0001_list',
#  'person100_bacteria_481_list',
#  'person171_bacteria_826_list',
#  'person108_bacteria_511_list',
#  'person34_virus_76_list',
#  'person100_bacteria_482_list',
#  'person136_bacteria_654_list',
#  'person147_bacteria_705_list',
#  'person61_virus_118_list',
#  'person157_bacteria_735_list',
#  'person1650_virus_2852_list',
#  'person11_virus_38_list',
#  'person109_bacteria_512_list',
#  'person133_bacteria_637_list',
#  'NORMAL2-IM-0359-0001_list',
#  'person83_bacteria_414_list',
#  'person1_virus_7_list',
#  'person37_virus_82_list',
#  'person141_bacteria_670_list',
#  'person144_bacteria_690_list',
#  'NORMAL2-IM-0272-0001_list',
#  'person91_bacteria_445_list',
#  'person119_bacteria_565_list',
#  'person103_bacteria_488_list',
#  'IM-0050-0001_list',
#  'person111_bacteria_536_list',
#  'NORMAL2-IM-0019-0001_list',
#  'person173_bacteria_831_list',
#  'person134_bacteria_643_list',
#  'person155_bacteria_731_list',
#  'person124_bacteria_592_list',
#  'person1650_virus_2854_list',
#  'person83_bacteria_409_list',
#  'person78_bacteria_381_list',
#  'person117_bacteria_556_list',
#  'person85_bacteria_421_list',
#  'person140_bacteria_667_list',
#  'person121_bacteria_576_list',
#  'person127_bacteria_602_list',
#  'person81_bacteria_398_list',
#  'person96_bacteria_465_list',
#  'person1644_virus_2844_list',
#  'person153_bacteria_725_list',
#  'person94_bacteria_458_list',
#  'person1612_virus_2797_list',
#  'person137_bacteria_655_list',
#  'person1610_virus_2793_list',
#  'person142_bacteria_683_list',
#  'person139_bacteria_664_list',
#  'person93_bacteria_453_list',
#  'person126_bacteria_598_list',
#  'person114_bacteria_545_list',
#  'person104_bacteria_491_list',
#  'person87_bacteria_433_list',
#  'person150_bacteria_716_list',
#  'person125_bacteria_595_list',
#  'person96_bacteria_466_list',
#  'person128_bacteria_608_list',
#  'person109_bacteria_526_list',
#  'person128_bacteria_607_list',
#  'person134_bacteria_644_list',
#  'person97_bacteria_468_list',
#  'person173_bacteria_830_list',
#  'person3_virus_17_list',
#  'person138_bacteria_659_list',
#  'person111_bacteria_533_list',
#  'NORMAL2-IM-0369-0001_list',
#  'NORMAL2-IM-0081-0001_list',
#  'person141_bacteria_678_list',
#  'person127_bacteria_603_list',
#  'person151_bacteria_718_list',
#  'person139_bacteria_661_list',
#  'person86_bacteria_429_list',
#  'person124_bacteria_590_list',
#  'person111_bacteria_537_list',
#  'person138_bacteria_657_list',
#  'person113_bacteria_540_list',
#  'person108_bacteria_504_list',
#  'person121_bacteria_580_list',
#  'person134_bacteria_642_list',
#  'person149_bacteria_713_list',
#  'person101_bacteria_485_list',
#  'person130_bacteria_626_list',
#  'person113_bacteria_542_list',
#  'person91_bacteria_449_list',
#  'person90_bacteria_443_list',
#  'person88_bacteria_438_list',
#  'NORMAL2-IM-0351-0001_list',
#  'person122_bacteria_585_list',
#  'person111_bacteria_534_list']




# Initiate plot figure 
plt.figure()
plt.xlabel("Step")
plt.ylabel("Score")
plt.title("Prediction score after each deletion")


# For each method
for method in mths:
# method = "InputXGradient"
    print(method)

    # Join path
    methodPath = os.path.join(path, method)

    # Create the numpy array where we will calculate the mean over the images
    values = np.zeros(1001)

    # Initial number of images 
    number = 0

    # For images in ... 
    # for each probabilities list of img
    for txt in os.listdir(methodPath):
    # for txt in cluster:

#         print(txt)

        # Avoid jupyter notebook checkpoints existing within a folder 
        if txt==".ipynb_checkpoints":
            continue
        
        # join path to the txt proba file
        txtPath = os.path.join(methodPath,txt)

        # try to open except error (for files not yet computed but existing in cluster images)
        try:
            file = open(txtPath, "r") 
        except FileNotFoundError:
            continue

        # Readline 
        listing = file.readline()
        value = listing.split(' ')

        # +1 image
        number+=1

        # Last element of value is '' which should not be read --> -1
        for v in range(len(value)-1):
            values[v]=values[v]+float(value[v])


    # Total number of images used for current method
    print("number = " + str(number))

    # Divide the total by the number of images to get the mean
    values = values/number

    # Plot the current method result over the mean of the images readen
    plt.plot(values, label=method)
    
# Legend and save figure
plt.legend()
# plt.show()
plt.savefig("Graph{}.png".format(cluster[0]))
