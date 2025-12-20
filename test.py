import cv2
import os.path as osp
import logging
import argparse

import options.options as option
import utils.util as util
from data import create_dataset, create_dataloader
from models import create_model

#### options
parser = argparse.ArgumentParser()
parser.add_argument('-opt', type=str, default='./options/test/huawei.yml', help='Path to options YMAL file.')
opt = option.parse(parser.parse_args().opt, is_train=False)
opt = option.dict_to_nonedict(opt)


def main():
    save_imgs = True
    model = create_model(opt)
    save_folder = './results/{}'.format(opt['name'])
    GT_folder = osp.join(save_folder, 'images/GT')
    output_folder = osp.join(save_folder, 'images/output')
    input_folder = osp.join(save_folder, 'images/input')

    x_LL3_gamma_folder = osp.join(save_folder, 'images/x_LL3_gamma')
    x_LLgt3_folder = osp.join(save_folder, 'images/x_LLgt3')
    gradient3_folder = osp.join(save_folder, 'images/gradient3')
    gradient3_gt_folder = osp.join(save_folder, 'images/gradient3_gt')
    x_l2_folder = osp.join(save_folder, 'images/x_l2')
    x_LLgt2_folder = osp.join(save_folder, 'images/x_LLgt2')
    x_l1_folder = osp.join(save_folder, 'images/x_l1')
    x_LLgt1_folder = osp.join(save_folder, 'images/x_LLgt1')


    util.mkdirs(save_folder)
    util.mkdirs(GT_folder)
    util.mkdirs(output_folder)
    util.mkdirs(input_folder)

    util.mkdirs(x_LL3_gamma_folder)
    util.mkdirs(x_LLgt3_folder)
    util.mkdirs(gradient3_folder)
    util.mkdirs(gradient3_gt_folder)
    util.mkdirs(x_l2_folder)
    util.mkdirs(x_LLgt2_folder)
    util.mkdirs(x_l1_folder)
    util.mkdirs(x_LLgt1_folder)

    print('mkdir finish')

    util.setup_logger('base', save_folder, 'test', level=logging.INFO, screen=True, tofile=True)
    logger = logging.getLogger('base')


    for phase, dataset_opt in opt['datasets'].items():
        val_set = create_dataset(dataset_opt)
        val_loader = create_dataloader(val_set, dataset_opt, opt, None)

        pbar = util.ProgressBar(len(val_loader))
        psnr_rlt = {}  # with border and center frames
        psnr_rlt_avg = {}
        psnr_total_avg = 0.

        ssim_rlt = {}  # with border and center frames
        ssim_rlt_avg = {}
        ssim_total_avg = 0.

        for val_data in val_loader:
            folder = val_data['folder'][0]
            idx_d = val_data['idx']
            if psnr_rlt.get(folder, None) is None:
                psnr_rlt[folder] = []

            if ssim_rlt.get(folder, None) is None:
                ssim_rlt[folder] = []
            model.feed_data(val_data)

            model.test()
            visuals = model.get_current_visuals()
            rlt_img = util.tensor2img(visuals['rlt'])  # uint8
            gt_img = util.tensor2img(visuals['GT'])  # uint8
            input_img = util.tensor2img(visuals['LQ'])
            x_LL3_gamma = util.tensor2img(visuals['x_LL3_gamma'])
            x_LLgt3 = util.tensor2img(visuals['x_LLgt3'])
            x_l2 = util.tensor2img(visuals['x_l2'])
            x_LLgt2 = util.tensor2img(visuals['x_LLgt2'])
            x_l1 = util.tensor2img(visuals['x_l1'])
            x_LLgt1 = util.tensor2img(visuals['x_LLgt1'])
            gradient3 = util.tensor2img(visuals['gradient3'])
            gradient3_gt = util.tensor2img(visuals['gradient3_gt'])
            if save_imgs:
                try:
                    tag = '{}'.format(val_data['folder']).split('[')[1].split(']')[0]
                    tag = tag.replace("'", "")
                    print(" ********* ", tag)
                    print(osp.join(output_folder, tag))
                    cv2.imwrite(osp.join(output_folder, tag), rlt_img)
                    cv2.imwrite(osp.join(GT_folder, tag), gt_img)
                    cv2.imwrite(osp.join(input_folder, tag), input_img)
                    cv2.imwrite(osp.join(x_LL3_gamma_folder, tag), x_LL3_gamma)
                    cv2.imwrite(osp.join(x_LLgt3_folder, tag), x_LLgt3)
                    cv2.imwrite(osp.join(x_l2_folder, tag), x_l2)
                    cv2.imwrite(osp.join(x_LLgt2_folder, tag), x_LLgt2)
                    cv2.imwrite(osp.join(x_l1_folder, tag), x_l1)
                    cv2.imwrite(osp.join(x_LLgt1_folder, tag), x_LLgt1)
                    cv2.imwrite(osp.join(gradient3_folder, tag), gradient3)
                    cv2.imwrite(osp.join(gradient3_gt_folder, tag), gradient3_gt)

                except Exception as e:
                    print(e)
                    import ipdb; ipdb.set_trace()

            # calculate PSNR
            psnr = util.calculate_psnr(rlt_img, gt_img)
            psnr_rlt[folder].append(psnr)

            ssim = util.calculate_ssim(rlt_img, gt_img)
            # ssim = 0
            ssim_rlt[folder].append(ssim)

            pbar.update('Test {} - {}'.format(folder, idx_d))
        for k, v in psnr_rlt.items():
            psnr_rlt_avg[k] = sum(v) / len(v)
            psnr_total_avg += psnr_rlt_avg[k]

        for k, v in ssim_rlt.items():
            ssim_rlt_avg[k] = sum(v) / len(v)
            ssim_total_avg += ssim_rlt_avg[k]

        psnr_total_avg /= len(psnr_rlt)
        ssim_total_avg /= len(ssim_rlt)
        log_s = '# Validation # PSNR: {:.4e}:'.format(psnr_total_avg)
        for k, v in psnr_rlt_avg.items():
            log_s += ' {}: {:.4e}'.format(k, v)
        logger.info(log_s)

        log_s = '# Validation # SSIM: {:.4e}:'.format(ssim_total_avg)
        for k, v in ssim_rlt_avg.items():
            log_s += ' {}: {:.4e}'.format(k, v)
        logger.info(log_s)

        psnr_all = 0
        psnr_count = 0
        for k, v in psnr_rlt.items():
            psnr_all += sum(v)
            psnr_count += len(v)
        psnr_all = psnr_all * 1.0 / psnr_count
        print(psnr_all)

        ssim_all = 0
        ssim_count = 0
        for k, v in ssim_rlt.items():
            ssim_all += sum(v)
            ssim_count += len(v)
        ssim_all = ssim_all * 1.0 / ssim_count
        print(ssim_all)


if __name__ == '__main__':
    main()
