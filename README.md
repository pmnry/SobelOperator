# SobelOperator
Implementation and testing of the Sobel Operator for image filtering
def histoImage(im):

    unique, counts = np.unique(im, return_counts=True)
    dic = dict(zip(unique, counts))
    df = pd.DataFrame.from_dict(dic, orient='index')
    df = df.rename(index=int, columns={0: "histVals"})

    return df

def otsu_thresh(h, w, hist):

    hist['wB'] = hist['histVals'].cumsum()
    hist['wF'] = (hist['histVals'][::-1].cumsum())

    hist['Product'] = hist.histVals*hist.index
    hist['mB'] = hist['Product'].cumsum()/hist['wB']
    hist['mF'] = (hist['Product'][::-1].cumsum() / hist['wF'][::-1])[::-1]

    # hist['VarBetw'] = hist['wB'] * hist['wF'] * (hist['mB'] - hist['mF']) ** 2
    hist['VarBetw'] = hist['wB'].iloc[:-1]*hist['wF'].iloc[1:]*(hist['mB'].iloc[:-1] - hist['mF'].iloc[1:])**2
    t = np.argmax(hist['VarBetw'])

    return t

def otsu_thresh_wrapper(im):
    height = im.shape[0]
    width = im.shape[1]
    histo = dict()

    if(len(im.shape)> 2):
        depth = im.shape[2]
        im_thresh = np.zeros((height, width, depth))

        for z in range(depth):
            histo[z] = histoImage(im[:, :, z])
            thresh_opt = otsu_thresh(height, width, histo[z])
            im_thresh[:, :, z] = im[:, :, z] >= thresh_opt

    else:
        depth = 1
        im_thresh = np.zeros((height, width))
        histo = histoImage(im)
        thresh_opt = otsu_thresh(height, width, histo)
        im_thresh = im>= thresh_opt

    return im_thresh

def img_apply(imag):
    height = imag.shape[0]
    width = imag.shape[1]

    if(len(imag.shape)>2):
        depth = imag.shape[2]
        res = np.zeros((height - 2, width - 2, depth))

        for z in range(depth):
            for i in range(1, height - 1):
                for j in range(1, width - 1):
                    res[i - 1, j - 1, z] = sobolop(np.asmatrix(imag[i - 1:i + 2, j - 1:j + 2, z]))
    else:
        res = np.zeros((height - 2, width - 2))
        for i in range(1, height - 1):
            for j in range(1, width - 1):
                res[i - 1, j - 1] = sobolop(np.asmatrix(imag[i - 1:i + 2, j - 1:j + 2]))


    return res
