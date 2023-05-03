import cmasher as cmr
from matplotlib import colors

# https://matplotlib.org/stable/_images/sphx_glr_colormaps_006.png
# https://matplotlib.org/stable/_images/sphx_glr_named_colors_003.png

def colour_mapper():
    colour_map_dict = {'scratch': colors.to_rgb('black'), 'joint': colors.to_rgb('gray')}
    cv_appr_list = ['lucir', 'eeil', 'bic', 'ssil', 'lwf', 'ewc', 'il2m', 'icarl', 'lwfgkd']
    net_appr_list = ['icarlp', 'chen2021', 'wu2022']
    naive_appr_list = ['jointft', 'finetuning', 'jointmem', 'backbonefreezing', 'freezing', 'backbonefreezingmem']
    appr_total_list = [cv_appr_list, net_appr_list, naive_appr_list]
    palette_list = ['tab10', 'Dark2',
                    ['navy', 'mediumvioletred', 'rebeccapurple', 'darkolivegreen', 'lime', 'forestgreen']]
    for i, appr_list in enumerate(appr_total_list):
        if isinstance(palette_list[i], str):
            sub_cmap = cmr.get_sub_cmap(palette_list[i], start=0, stop=1, N=None).colors
            for j, appr in enumerate(appr_list):
                colour_map_dict[appr] = sub_cmap[j]
        else:
            for appr, color in zip(appr_list, palette_list[i]):
                colour_map_dict[appr] = colors.to_rgb(color)
    colour_map_dict['_'] = (.0, .0, .0)
    return colour_map_dict


if __name__ == "__main__":
    cmap_dict = colour_mapper()
