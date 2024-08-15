from captum.attr import Saliency, IntegratedGradients, NoiseTunnel, GradientShap, Lime
import torch
from explainable_ai_image_measures import Measures      # Summary statistics IAUC, DAUC and IROF
import gc
import numpy as np


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NR_VAL_IMAGES_TO_TEST = 128
SUPERPIXEL_SIZE = 8


class AverageMeter(object):
    """Computes and stores the average and current value, and uses a fmt string to format values"""
    def __init__(self, name, fmt=":f"):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = "{name} {val" + self.fmt + "} ({avg" + self.fmt + "})"
        return fmtstr.format(**self.__dict__)


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


def generate_attributions(model, images, labels, batch_size, baselines):
    attributions_list = []

    # Split images and labels into batches
    num_batches = (len(images) + batch_size - 1) // batch_size

    for i in range(num_batches):
        batch_images = images[i*batch_size:(i+1)*batch_size]
        batch_labels = labels[i*batch_size:(i+1)*batch_size]

        rnd = torch.rand(batch_images.shape, device=device)

        sal = Saliency(model).attribute(batch_images, target=batch_labels)
        
        nt = NoiseTunnel(Saliency(model)).attribute(batch_images, target=batch_labels, nt_type="smoothgrad", nt_samples_batch_size=batch_size)
        
        # baselines = next(iter(val_acc_loader))[0].cuda()
        shap = GradientShap(model).attribute(batch_images, baselines, target=batch_labels)

        num_superpixels = 128 // SUPERPIXEL_SIZE
        feature_mask = torch.zeros((128, 128), dtype=torch.int).to(batch_images.device)
        value = 0
        for j in range(num_superpixels):
            for k in range(num_superpixels):
                feature_mask[j*SUPERPIXEL_SIZE:(j+1)*SUPERPIXEL_SIZE, k*SUPERPIXEL_SIZE:(k+1)*SUPERPIXEL_SIZE] = value
                value += 1
        lime = Lime(model).attribute(batch_images, feature_mask=feature_mask, target=batch_labels)

        ig = IntegratedGradients(model).attribute(batch_images, target=batch_labels)

        batch_attributions = torch.stack([rnd, sal, nt, shap, lime, ig])
        attributions_list.append(batch_attributions)

        sal = None
        nt = None
        shap = None
        lime = None
        ig = None
        gc.collect()
        torch.cuda.empty_cache()

    temp = torch.norm(torch.cat(attributions_list, dim=1), 2, dim=2).permute(1, 0, 2, 3)

    # attributions = None
    attributions_list = None
    gc.collect()
    torch.cuda.empty_cache()

    return temp


def sum_explanations(model, loader, attribution_methods):
    """
    Computes summary scores for a set of inputs run on a given model averaged over the inputs.
    model: model to run explanations for,
    loader: enumeratable image loader for set of images to run summary statistics for attributive methods for supplied model
    # inputs: Set of inputs to run explanation methods on,
    # targets: Target classes of inputs for which to explain model,
    attribution_methods: The explanation methods to run in a dictionary. 
                         - Keys are explanation methods names
                         - Values are explanation method ".attribute" function. If the ".attribute" has optional or necessary extra arguments,
                           a custom function with these prespecified has to be supplied (eg 
                           def customGradientShap(input, labels):
                             GradientShap(model).attribute(input, baselines, target=labels) # for a prespecified baselines)
    """
    model.eval()
    summed_attributions = dict()
    for name in attribution_methods.keys():
        summed_attributions[name] = 0
    # Explaining instances is a lot more expensive than simple inference, so we limit the total number of instances explained at each early stop with this.
    counter = 0

    # We drop the provided "correct" labels because we are interested in evaluating the explanation methods' effectiveness in explaining the model's perceived class of each input.
    for i, (images, _) in enumerate(loader):
        counter += len(images)
        if(counter >= NR_VAL_IMAGES_TO_TEST): break

        images = images.to(device)
        labels = model(images).argmax(dim=1)

        for name, func in attribution_methods.items():
            summed_attributions[name] = summed_attributions[name] + torch.sum(torch.norm(func(images, labels=labels), 2, dim=1))

    # Explicitly clear memory as per: https://stackoverflow.com/questions/57858433/how-to-clear-gpu-memory-after-pytorch-model-training-without-restarting-kernel
    # This is performed such that maximum memory is available for training outside this function.
    gc.collect()                # Python thing
    torch.cuda.empty_cache()    # PyTorch thing

    if "Vanillagrad" in attribution_methods:
        print("sum_explanations->Vanillagrad: ", summed_attributions["Vanillagrad"])
    elif "IntegratedGradients" in attribution_methods:
        print("sum_explanations->IntegratedGradients: ", summed_attributions["IntegratedGradients"])

    for name, _ in attribution_methods.items():
        summed_attributions[name] = summed_attributions[name] / NR_VAL_IMAGES_TO_TEST

    return summed_attributions


def average_summary_scores(model, loader, 
        summary_batch_size,
        attribution_methods,
        summary_stats = ["IAUC"]):
    """
    Computes summary scores for a set of inputs run on a given model averaged over the inputs.
    model: model to run explanations for,
    loader: enumeratable image loader for set of images to run summary statistics for attributive methods for supplied model
    # inputs: Set of inputs to run explanation methods on,
    # targets: Target classes of inputs for which to explain model,
    summary_batch_size: The batch size to use when calculating explanations and generating summary scores. Recommended
                            to use much smaller batch size than when training model because memory usage of some explanation
                            methods is immense.
    attribution_methods: The explanation methods to run in a dictionary. 
                         - Keys are explanation methods names
                         - Values are explanation method ".attribute" function. If the ".attribute" has optional or necessary extra arguments,
                           a custom function with these prespecified has to be supplied (eg 
                           def customGradientShap(input, labels):
                             GradientShap(model).attribute(input, baselines, target=labels) # for a prespecified baselines)
    summary_stats: The summary statistics to use, the explainable_ai_image_measures library supports IAUC, DAUC and IROF 
    """
    model.eval()
    IAUC = False
    DAUC = False
    IROF = False
    if "IAUC" in summary_stats: IAUC = True
    if "DAUC" in summary_stats: DAUC = True
    if "IROF" in summary_stats: IROF = True
    measure = Measures(model, batch_size=summary_batch_size, pixel_package_size=50, normalize=False, clip01=False)

    method_names = list(attribution_methods.keys())
    summed_scores = dict()
    for stat in summary_stats:
        summed_scores[stat] = dict()
        for name in method_names:
            summed_scores[stat][name] = 0

    # Explaining instances is a lot more expensive than simple inference, so we limit the total number of instances explained at each early stop with this.
    counter = 0

    # We drop the provided "correct" labels because we are interested in evaluating the explanation methods' effectiveness in explaining the model's perceived class of each input.
    for i, (images, _) in enumerate(loader):
        counter += len(images)
        if(counter >= NR_VAL_IMAGES_TO_TEST): break

        images = images.to(device)
        labels = model(images).argmax(dim=1)

        preprocessed_attributions = torch.norm(torch.stack([func(images, labels=labels) for func in attribution_methods.values()]), 2, dim=2).permute(1, 0, 2, 3)


        scoring_results = measure.compute_batch(images, preprocessed_attributions, labels, IAUC=IAUC, DAUC=DAUC, IROF=IROF)

        # TODO: maybe do batched implementation? (can we depend on scoring_results to always return the summary stats in the same order?)
        for attr_ind in range(len(attribution_methods)):
            attr_name = method_names[attr_ind]
            for measure_id, measure_title in enumerate(scoring_results.keys()):
                # print(scoring_results[measure_title][0])
                score = scoring_results[measure_title][0][:, attr_ind]      # Grab one summary stat score for all images in current image batch and for specified attribution_method
                score = torch.sum(score, dim=0)                             # Sum accross the batch of images
                # Add current batch of images' scores to the summed_scores for the current summary stat and attribution_method
                summed_scores[measure_title][attr_name] = summed_scores[measure_title][attr_name] + score


    # Explicitly clear memory for these heavy variables as per: https://stackoverflow.com/questions/57858433/how-to-clear-gpu-memory-after-pytorch-model-training-without-restarting-kernel
    # This is performed such that maximum memory is available for training outside this function.
    measure = None
    # attributions_list = None
    preprocessed_attributions = None
    scoring_results = None
    gc.collect()                # Python thing
    torch.cuda.empty_cache()    # PyTorch thing

    average_scores = dict()
    for stat in summary_stats:
        average_scores[stat] = dict()
        for attribution_name, _ in attribution_methods.items():
            average_scores[stat][attribution_name] = summed_scores[stat][attribution_name] / NR_VAL_IMAGES_TO_TEST

    return average_scores


def running_stats(y, window_size):
    avg = np.convolve(y, np.ones(window_size)/window_size, mode="valid")
    squared_diff = np.convolve((y - np.convolve(y, np.ones(window_size)/window_size, mode="same"))**2, np.ones(window_size)/window_size, mode="valid")
    std = np.sqrt(squared_diff)
    return avg, std
