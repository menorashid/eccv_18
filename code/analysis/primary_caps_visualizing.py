import sys
sys.path.append('./')
from helpers import util, visualize, receptive_field
import os
import numpy as np
import scipy.misc
import sklearn.metrics
import glob
import multiprocessing
import sklearn.cluster
import sklearn.preprocessing
import sklearn.decomposition
import matplotlib.pyplot as plt
import math

dir_server = '/disk3'
str_replace = ['..',os.path.join(dir_server,'maheen_data/eccv_18')]
click_str = 'http://vision3.idav.ucdavis.edu:1000'


def get_caps_compiled(routed = False):
    

    model_name = 'khorrami_capsule_7_3_bigclass'
    route_iter = 3
    pre_pend = 'ck_96_train_test_files_'
    strs_append = '_reconstruct_True_True_all_aug_margin_False_wdecay_0_600_exp_0.96_350_1e-06_0.001_0.001_0.001'
    model_num = 599
    split_num = 4
    
    out_dir_meta = os.path.join('../experiments',model_name+str(route_iter))
    out_dir_train =  os.path.join(out_dir_meta,pre_pend+str(split_num)+strs_append)
    train_pre =  os.path.join('../data/ck_96','train_test_files')
    test_file =  os.path.join(train_pre,'train_'+str(split_num)+'.txt')

    caps_dir = os.path.join(out_dir_train, 'save_primary_caps_train_data_'+str(model_num))
    # caps_files = glob.glob(os.path.join(caps_dir,'*.npy'))
    caps_files = [os.path.join(caps_dir,str(num)+'.npy') for num in range(10)]
    if routed:
        route_files = [os.path.join(caps_dir,str(num)+'_routes.npy') for num in range(10)]
        routes = []
        for route_file in route_files:
            routes.append(np.load(route_file))
        routes = np.concatenate(routes,1)



    convnet =   [[5,1,2],[2,2,0],[5,1,2],[2,2,0],[7,3,0]]
    imsize = 96

    caps = []
    for caps_file in caps_files:
        caps.append(np.load(caps_file))
    print len(caps)
    print caps[0].shape
    caps = np.concatenate(caps,0)
    print caps.shape
    if routed:
        return caps, test_file, convnet, imsize, routes
    else:
        return caps, test_file, convnet, imsize

def save_ims(mags,filter_num,x,y,test_im,out_dir_curr,convnet,imsize,rewrite = False):
    vec_rel = mags[:,filter_num,x,y]
    print vec_rel.shape
    idx_sort = np.argsort(vec_rel)[::-1]
    print vec_rel[idx_sort[0]]
    print vec_rel[idx_sort[-1]]

    im_row = []
    caption_row =[]
    for idx_idx,idx_curr in enumerate(idx_sort):
        out_file_curr = os.path.join(out_dir_curr,str(idx_idx)+'.jpg')
        if not os.path.exists(out_file_curr) or rewrite:
            im_curr = test_im[idx_curr]
            rec_field, center = receptive_field.get_receptive_field(convnet,imsize,len(convnet)-1, x,y)
            center = [int(round(val)) for val in center]
            range_x = [max(0,center[0]-rec_field/2),min(imsize,center[0]+rec_field/2)]
            range_y = [max(0,center[1]-rec_field/2),min(imsize,center[1]+rec_field/2)]
            im_curr = im_curr[range_y[0]:range_y[1],range_x[0]:range_x[1]]
            # print out_file_curr
            # raw_input()
            scipy.misc.imsave(out_file_curr,im_curr)
        im_row.append(out_file_curr)
        caption_row.append('%d %.4f' % (idx_idx,vec_rel[idx_curr]))
    return im_row,caption_row


def k_means(caps,num_clusters, filter_num,x,y,test_im,out_dir_curr,out_file_html,convnet,imsize,rewrite = False):
    vec_rel_org = caps[:,filter_num,x,y,:]
    k_meaner = sklearn.cluster.KMeans(n_clusters=num_clusters)
    vec_rel = sklearn.preprocessing.normalize(vec_rel_org,axis=0) #feature normalize
    vec_rel = vec_rel_org

    bins = k_meaner.fit_predict(vec_rel)
    print bins
    for val in np.unique(bins):
        print val, np.sum(bins==val)

    im_row = [[] for idx in range(num_clusters)]
    caption_row = [[] for idx in range(num_clusters)]
    for idx_idx,bin_curr in enumerate(bins):
        out_file_curr = os.path.join(out_dir_curr,str(idx_idx)+'.jpg')
        # if not os.path.exists(out_file_curr) or rewrite:
        im_curr = test_im[idx_idx]
        rec_field, center = receptive_field.get_receptive_field(convnet,imsize,len(convnet)-1, x,y)
        center = [int(round(val)) for val in center]
        range_x = [max(0,center[0]-rec_field/2),min(imsize,center[0]+rec_field/2)]
        range_y = [max(0,center[1]-rec_field/2),min(imsize,center[1]+rec_field/2)]
        im_curr = im_curr[range_y[0]:range_y[1],range_x[0]:range_x[1]]
        # print out_file_curr
        # raw_input()
        scipy.misc.imsave(out_file_curr,im_curr)
        im_row[bin_curr].append(util.getRelPath(out_file_curr,dir_server))
        # print bin_curr,np.linalg.norm(vec_rel_org[idx_idx])
        caption_row[bin_curr].append('%d %.4f' % (bin_curr,np.linalg.norm(vec_rel_org[idx_idx])))

    # out_file_html = out_dir_curr+'.html'
    visualize.writeHTML(out_file_html,im_row,caption_row,40,40)
    print out_file_html
    return im_row, caption_row




def pca(caps,num_clusters, filter_num,x,y,test_im,out_dir_curr,out_file_html,convnet,imsize,rewrite = False):
    vec_rel = caps[:,filter_num,x,y,:]
    # pca = sklearn.decomposition.PCA(n_components=8, whiten = True)
    # vec_rel = sklearn.preprocessing.normalize(vec_rel_org,axis=0) #feature normalize
    # pca.fit(vec_rel_org)
    # print pca.explained_variance_ratio_  , np.sum(pca.explained_variance_ratio_)
    # vec_rel = pca.transform(vec_rel_org)
    # print vec_rel.shape
    im_rows = []
    caption_rows = []
    for vec_curr_idx in range(vec_rel.shape[1]): 
        directions = vec_rel[:,vec_curr_idx]
        # directions = vec_rel/np.linalg.norm(vec_rel,axis=1,keepdims=True)
        # directions = np.arctan(directions[:,0]/directions[:,1])
        # print np.min(directions), np.max(directions)
        idx_sort = np.argsort(directions)

        # print vec_rel.shape
        

        # plt.figure()
        # plt.plot(directions[:,0],directions[:,1],'*b')
        # plt.savefig(out_dir_curr+'.jpg')
        # plt.close()
        # raw_input()

        im_row = []
        # [] for idx in range(num_clusters)]
        caption_row = []
        # [] for idx in range(num_clusters)]
        for idx_idx,idx_curr in enumerate(idx_sort):
            out_file_curr = os.path.join(out_dir_curr,str(idx_idx)+'.jpg')
            # if not os.path.exists(out_file_curr) or rewrite:
            im_curr = test_im[idx_curr]
            rec_field, center = receptive_field.get_receptive_field(convnet,imsize,len(convnet)-1, x,y)
            center = [int(round(val)) for val in center]
            range_x = [max(0,center[0]-rec_field/2),min(imsize,center[0]+rec_field/2)]
            range_y = [max(0,center[1]-rec_field/2),min(imsize,center[1]+rec_field/2)]
            im_curr = im_curr[range_y[0]:range_y[1],range_x[0]:range_x[1]]
            # print out_file_curr
            # raw_input()
            scipy.misc.imsave(out_file_curr,im_curr)
            im_row.append(util.getRelPath(out_file_curr,dir_server))
            # [bin_curr].append(util.getRelPath(out_file_curr,dir_server))
            # print bin_curr,np.linalg.norm(vec_rel_org[idx_idx])
            caption_row.append('%d %.2f' % (idx_curr,directions[idx_curr]))

        im_rows.append(im_row)
        caption_rows.append(caption_row)
    # out_file_html = out_dir_curr+'.html'
    visualize.writeHTML(out_file_html,im_rows,caption_rows,40,40)
    print out_file_html


    # k_meaner = sklearn.cluster.KMeans(n_clusters=num_clusters)
    # vec_rel = sklearn.preprocessing.normalize(vec_rel_org,axis=0) #feature normalize
    # vec_rel = vec_rel_org

    # bins = k_meaner.fit_predict(vec_rel)
    # print bins
    # for val in np.unique(bins):
    #     print val, np.sum(bins==val)

    # im_row = [[] for idx in range(num_clusters)]
    # caption_row = [[] for idx in range(num_clusters)]
    # for idx_idx,bin_curr in enumerate(bins):
    #     out_file_curr = os.path.join(out_dir_curr,str(idx_idx)+'.jpg')
    #     # if not os.path.exists(out_file_curr) or rewrite:
    #     im_curr = test_im[idx_idx]
    #     rec_field, center = receptive_field.get_receptive_field(convnet,imsize,len(convnet)-1, x,y)
    #     center = [int(round(val)) for val in center]
    #     range_x = [max(0,center[0]-rec_field/2),min(imsize,center[0]+rec_field/2)]
    #     range_y = [max(0,center[1]-rec_field/2),min(imsize,center[1]+rec_field/2)]
    #     im_curr = im_curr[range_y[0]:range_y[1],range_x[0]:range_x[1]]
    #     # print out_file_curr
    #     # raw_input()
    #     scipy.misc.imsave(out_file_curr,im_curr)
    #     im_row[bin_curr].append(util.getRelPath(out_file_curr,dir_server))
    #     # print bin_curr,np.linalg.norm(vec_rel_org[idx_idx])
    #     caption_row[bin_curr].append('%d %.4f' % (bin_curr,np.linalg.norm(vec_rel_org[idx_idx])))

    # # out_file_html = out_dir_curr+'.html'
    # visualize.writeHTML(out_file_html,im_row,caption_row,40,40)
    # print out_file_html
    # return im_row, caption_row

    



def script_viz_k_means():
    out_dir_htmls = '../experiments/figures/primary_caps_viz_pca'.replace(str_replace[0],str_replace[1])
    util.mkdir(out_dir_htmls)
    out_dir_im = os.path.join(out_dir_htmls,'im')
    util.mkdir(out_dir_im)


    caps, test_file, convnet, imsize  = get_caps_compiled()
    num_clusters = 32


    # arr_vals = [(x,y,filter_num) for x in range(6) for y in range(6) for filter_num in range(32)]
    arr_vals = [(x,y,filter_num) for x in [3] for y in [5] for filter_num in [3]]

    test_im = [scipy.misc.imread(line_curr.split(' ')[0]) for line_curr in util.readLinesFromFile(test_file)]
    print len(test_im)
    print test_im[0].shape

    for x,y,filter_num in arr_vals:
        out_dir_curr = os.path.join(out_dir_im,str(x)+'_'+str(y)+'_'+str(filter_num))
        util.mkdir(out_dir_curr)
        out_file_html = os.path.join(out_dir_htmls,str(x)+'_'+str(y)+'_'+str(filter_num)+'.html')
        # if os.path.exists(out_file_html):
        #     continue
        pca(caps,num_clusters, filter_num,x,y,test_im,out_dir_curr,out_file_html,convnet,imsize,rewrite = False)
        # break
    visualize.writeHTMLForFolder(out_dir_im)


def script_viz_mag():
    
    out_dir_htmls = '../experiments/figures/primary_caps_viz'.replace(str_replace[0],str_replace[1])
    util.mkdir(out_dir_htmls)
    out_dir_im = os.path.join(out_dir_htmls,'im')
    util.mkdir(out_dir_im)

    caps, test_file, convnet, imsize  = get_caps_compiled()
    mags = np.linalg.norm(caps,  axis = 4)
    print mags.shape
    print np.min(mags), np.max(mags)

    test_im = [scipy.misc.imread(line_curr.split(' ')[0]) for line_curr in util.readLinesFromFile(test_file)]
    print len(test_im)
    print test_im[0].shape

    for x in range(mags.shape[2]):
            for y in range(mags.shape[3]):
                out_file_html = os.path.join(out_dir_htmls,str(x)+'_'+str(y)+'.html')
                ims_html = []
                captions_html = []

                for filter_num in range(mags.shape[1]):
                    out_dir_curr = os.path.join(out_dir_im,str(x)+'_'+str(y)+'_'+str(filter_num))
                    util.mkdir(out_dir_curr)
                
                    im_row,caption_row = save_ims(mags,filter_num,x,y,test_im,out_dir_curr, convnet, imsize)
                    im_row = [util.getRelPath(im_curr,dir_server) for im_curr in im_row]
                    # caption_row = [os.path.split(im_curr)[1][:-4] for im_curr in im_row]
                    ims_html.append(im_row[:10]+im_row[-10:])
                    captions_html.append(caption_row[:10]+caption_row[-10:])

                visualize.writeHTML(out_file_html,ims_html,captions_html,40,40)

def save_all_patches():
    out_dir = '../experiments/figures/primary_caps_viz/im_all_patches/train'
    util.makedirs(out_dir)
    _, test_file, convnet, imsize  = get_caps_compiled()
    test_im = [scipy.misc.imread(line_curr.split(' ')[0]) for line_curr in util.readLinesFromFile(test_file)]

    for idx_test_im_curr,im_curr in enumerate(test_im):
        for x in range(6):
            for y in range(6):

                out_file_curr = os.path.join(out_dir,'_'.join([str(val) for val in [idx_test_im_curr,x,y]])+'.jpg')
                print out_file_curr
                rec_field, center = receptive_field.get_receptive_field(convnet,imsize,len(convnet)-1, x,y)
                center = [int(round(val)) for val in center]
                range_x = [max(0,center[0]-rec_field/2),min(imsize,center[0]+rec_field/2)]
                range_y = [max(0,center[1]-rec_field/2),min(imsize,center[1]+rec_field/2)]
                patch = im_curr[range_y[0]:range_y[1],range_x[0]:range_x[1]]
                # print out_file_curr
                # raw_input()
                scipy.misc.imsave(out_file_curr,patch)



def script_view_all_patches_sorted():

    out_dir_meta = '../experiments/figures/primary_caps_viz'.replace(str_replace[0],str_replace[1])
    out_dir_im = os.path.join(out_dir_meta,'im_all_patches/train')

    caps, test_file, convnet, imsize  = get_caps_compiled(routed= False)
    mags = np.linalg.norm(caps,axis = 4)

    mags_org = mags
    print 'mags_org.shape',mags_org.shape

    mags = np.transpose(mags,(0,2,3,1))
    print mags.shape
    mags = np.reshape(mags,(mags.shape[0]*mags.shape[1]*mags.shape[2],mags.shape[3]))
    print mags.shape
    idx_helper = range(mags.shape[0])
    print len(idx_helper)
    idx_helper = np.reshape(idx_helper,(caps.shape[0],caps.shape[2],caps.shape[3]))
    print idx_helper.shape


    num_to_keep = 100
    print 'mags_org.shape',mags_org.shape


    out_file_html = os.path.join(out_dir_meta,'mag_sorted.html')

    im_rows = []
    caption_rows = []

    for filt_num in range(mags.shape[1]):
        im_row = []
        caption_row =[]
        mag_curr = mags[:,filt_num]
        print np.min(mag_curr), np.max(mag_curr)
        idx_sort = list(np.argsort(mag_curr)[::-1])
        idx_sort = idx_sort[:num_to_keep]+idx_sort[-num_to_keep:]

        sorted_mag_curr = mag_curr[idx_sort]
        # print sorted_mag_curr[0],sorted_mag_curr[-1]
        # raw_input()
        

        for idx_idx, idx_curr in enumerate(idx_sort):
            arg_multi_dim = np.where(idx_helper==idx_curr)
            arg_multi_dim = [arr[0] for arr in arg_multi_dim]
            # print arg_multi_dim
            # if arg_multi_dim[1]==0 or arg_multi_dim[1]==5 or arg_multi_dim[2]==0 or arg_multi_dim[2]==5:
            #     continue
            # arg_multi_dim = [arg_multi_dim[0],max(arg_multi_dim[2],1),max(arg_multi_dim[1],1)]
            file_curr = os.path.join(out_dir_im,'_'.join([str(val) for val in arg_multi_dim])+'.jpg')
            assert os.path.exists(file_curr)
            im_row.append(util.getRelPath(file_curr,dir_server))
            caption_row.append('%d %.4f' % (idx_idx, sorted_mag_curr[idx_idx]))
            # if len(im_row)==num_to_keep:
            #     break
        im_rows.append(im_row)
        caption_rows.append(caption_row)

    visualize.writeHTML(out_file_html,im_rows,caption_rows,40,40)
    print out_file_html.replace(dir_server,click_str)


def script_view_clusters_high_mag(mag_sorted = False, mag_percent = 0.25):
    out_dir_meta = '../experiments/figures/primary_caps_viz'.replace(str_replace[0],str_replace[1])
    out_dir_im = os.path.join(out_dir_meta,'im_all_patches/train')
    
    out_dir_meta = '../experiments/figures/primary_caps_viz_clusters_top_'+str(mag_percent).replace(str_replace[0],str_replace[1])
    util.mkdir(out_dir_meta)

    caps, test_file, convnet, imsize  = get_caps_compiled(routed = False)
    mags = np.linalg.norm(caps,axis = 4)
    mags_org = mags
    mags = np.transpose(mags,(0,2,3,1))
    mags = np.reshape(mags,(mags.shape[0]*mags.shape[1]*mags.shape[2],mags.shape[3]))
    
    print mags.shape


    # print test_file
    gt_class = [int(line_curr.split(' ')[1]) for line_curr in util.readLinesFromFile(test_file)]

    caps_org = np.array(caps)

    caps = np.transpose(caps,(0,2,3,1,4))
    # print caps.shape
    caps = np.reshape(caps,(caps.shape[0]*caps.shape[1]*caps.shape[2],caps.shape[3],caps.shape[4]))
    print caps.shape
    # print mags.shape
    idx_helper = range(caps.shape[0])
    # print len(idx_helper)
    idx_helper = np.reshape(idx_helper,(caps_org.shape[0],caps_org.shape[2],caps_org.shape[3]))
    # print idx_helper.shape



    num_to_keep = 100
    num_clusters = 32

    for filt_num in range(caps.shape[1]):
        if mag_sorted:
            out_file_html = os.path.join(out_dir_meta,str(filt_num)+'_mag_sorted.html')
        else:
            out_file_html = os.path.join(out_dir_meta,str(filt_num)+'.html')
            
        im_rows = []
        caption_rows = []
        
        caps_curr = caps[:,filt_num]    
        mags_curr = mags[:,filt_num]
        print caps_curr.shape
        # get top mags
        idx_sort_mags = np.argsort(mags_curr)[::-1]
        num_to_keep_cluster = int(math.floor(len(mags_curr)*mag_percent))
        idx_to_cluster = idx_sort_mags[:num_to_keep_cluster]

        caps_curr_to_cluster = caps_curr[idx_to_cluster]
        mags_curr_to_cluster = mags_curr[idx_to_cluster]

        print caps_curr_to_cluster.shape
        print mags_curr_to_cluster.shape


        # print num_to_keep_cluster,idx_to_cluster.shape
        # print num_to_keep_cluster
        # print mags_curr[idx_sort_mags[0]],mags_curr[idx_sort_mags[-1]]
        # print np.min(mags_curr),np.max(mags_curr)
        # raw_input()



        k_meaner = sklearn.cluster.KMeans(n_clusters=num_clusters)
        vec_rel = sklearn.preprocessing.normalize(caps_curr_to_cluster,axis = 1)
        
        k_meaner.fit(np.random.permutation(vec_rel))
        cluster_centers = k_meaner.cluster_centers_
        
        cluster_belongings = k_meaner.predict(vec_rel)
        # print cluster_centers,cluster_centers.shape

        for idx_cluster_center,cluster_center in enumerate(cluster_centers):
            if mag_sorted:
                idx_rel = np.where(cluster_belongings == idx_cluster_center)[0]
                # print idx_rel.shape
                # print idx_rel[:10]
                mag_rel = mags_curr_to_cluster[idx_rel]
                idx_sort = np.argsort(mag_rel)[::-1]
                
                idx_sort = list(idx_to_cluster[idx_rel[idx_sort]])

            # print idx_sort[:10]
            # raw_input()
            else:            
                cluster_center = cluster_center[np.newaxis,:]
            #     # print (vec_rel-cluster_center).shape
                dist = np.linalg.norm(vec_rel-cluster_center,axis = 1)
            #     # print dist.shape
            #     # print mags.shape
            #     # raw_input()
                idx_sort = list(np.argsort(dist))
                idx_sort = idx_to_cluster[idx_sort]

            idx_sort = idx_sort[:num_to_keep]
            # +idx_sort[-num_to_keep:]

            im_row = []
            caption_row =[]

            for idx_idx, idx_curr in enumerate(idx_sort):
                arg_multi_dim = np.where(idx_helper==idx_curr)
                arg_multi_dim = [arr[0] for arr in arg_multi_dim]
                
                file_curr = os.path.join(out_dir_im,'_'.join([str(val) for val in arg_multi_dim])+'.jpg')
                assert os.path.exists(file_curr)
                im_row.append(util.getRelPath(file_curr,dir_server))
                caption_row.append('%d %.4f' %(idx_idx,mags_curr[idx_curr]))
                    # str(idx_idx)+' '+str(filt_num))
                
            im_rows.append(im_row)
            caption_rows.append(caption_row)

        visualize.writeHTML(out_file_html,im_rows,caption_rows,40,40)
        print out_file_html.replace(dir_server,click_str)


def script_view_clusters(routed = False,mag_sorted = True):
    out_dir_meta = '../experiments/figures/primary_caps_viz'.replace(str_replace[0],str_replace[1])
    out_dir_im = os.path.join(out_dir_meta,'im_all_patches/train')
    
    out_dir_meta = '../experiments/figures/primary_caps_viz_clusters'.replace(str_replace[0],str_replace[1])
    util.mkdir(out_dir_meta)

    caps, test_file, convnet, imsize, routes  = get_caps_compiled(routed= True)
    mags = np.linalg.norm(caps,axis = 4)
    mags_org = mags
    # print 'mags_org.shape',mags_org.shape
    mags = np.transpose(mags,(0,2,3,1))
    # print mags.shape
    mags = np.reshape(mags,(mags.shape[0]*mags.shape[1]*mags.shape[2],mags.shape[3]))
    # print mags.shape

    
    # print routes.shape

    # print test_file
    gt_class = [int(line_curr.split(' ')[1]) for line_curr in util.readLinesFromFile(test_file)]

    routes_gt = routes[gt_class,range(routes.shape[1])].squeeze()
    mag_routes = np.linalg.norm(routes_gt,axis = 2)
    # np.sum(routes_gt,axis=2)
    # 
    mag_routes = np.reshape(mag_routes,(mag_routes.shape[0],32,6,6,1))
    
    # print np.min(mag_routes),np.max(mag_routes)
    # print mag_routes.shape
    # print caps.shape
    if routed:
        caps = caps*mag_routes


    caps_org = np.array(caps)

    caps = np.transpose(caps,(0,2,3,1,4))
    # print caps.shape
    caps = np.reshape(caps,(caps.shape[0]*caps.shape[1]*caps.shape[2],caps.shape[3],caps.shape[4]))
    # print caps.shape
    # print mags.shape
    idx_helper = range(caps.shape[0])
    # print len(idx_helper)
    idx_helper = np.reshape(idx_helper,(caps_org.shape[0],caps_org.shape[2],caps_org.shape[3]))
    # print idx_helper.shape


    num_to_keep = 100
    num_clusters = 32

    for filt_num in range(caps.shape[1]):
        if mag_sorted:
            out_file_html = os.path.join(out_dir_meta,str(filt_num)+'_mag_sorted.html')
        elif routed:
            out_file_html = os.path.join(out_dir_meta,str(filt_num)+'_route_weighted.html')
        else:
            out_file_html = os.path.join(out_dir_meta,str(filt_num)+'.html')
            
        im_rows = []
        caption_rows = []
        
        caps_curr = caps[:,filt_num]    
        mags_curr = mags[:,filt_num]

        k_meaner = sklearn.cluster.KMeans(n_clusters=num_clusters)
        vec_rel = sklearn.preprocessing.normalize(caps_curr,axis = 1)
        # sklearn.preprocessing.normalize(sklearn.preprocessing.normalize(caps_curr,axis=0),axis=1) #feature normalize
        # print 'vec_rel.shape',vec_rel.shape
        print vec_rel.shape
        # numpy.random.permutation(x)
        k_meaner.fit(np.random.permutation(vec_rel))
        cluster_centers = k_meaner.cluster_centers_
        print cluster_centers.shape
        cluster_belongings = k_meaner.predict(vec_rel)
        # print cluster_centers,cluster_centers.shape

        for idx_cluster_center,cluster_center in enumerate(cluster_centers):
            if mag_sorted:
                idx_rel = np.where(cluster_belongings == idx_cluster_center)[0]
                # print idx_rel.shape
                # print idx_rel[:10]
                mag_rel = mags_curr[idx_rel]
                idx_sort = np.argsort(mag_rel)[::-1]
                idx_sort = list(idx_rel[idx_sort])
                # print idx_sort[:10]
                # raw_input()
            else:            
                cluster_center = cluster_center[np.newaxis,:]
                # print (vec_rel-cluster_center).shape
                dist = np.linalg.norm(vec_rel-cluster_center,axis = 1)
                # print dist.shape
                # print mags.shape
                # raw_input()
                idx_sort = list(np.argsort(dist))

            idx_sort = idx_sort[:num_to_keep]+idx_sort[-num_to_keep:]

            im_row = []
            caption_row =[]

            for idx_idx, idx_curr in enumerate(idx_sort):
                arg_multi_dim = np.where(idx_helper==idx_curr)
                arg_multi_dim = [arr[0] for arr in arg_multi_dim]
                
                file_curr = os.path.join(out_dir_im,'_'.join([str(val) for val in arg_multi_dim])+'.jpg')
                assert os.path.exists(file_curr)
                im_row.append(util.getRelPath(file_curr,dir_server))
                caption_row.append('%d %.4f' %(idx_idx,mags_curr[idx_curr]))
                    # str(idx_idx)+' '+str(filt_num))
                
            im_rows.append(im_row)
            caption_rows.append(caption_row)

        visualize.writeHTML(out_file_html,im_rows,caption_rows,40,40)
        print out_file_html.replace(dir_server,click_str)
 

def script_view_clusters(routed = False,mag_sorted = True):
    out_dir_meta = '../experiments/figures/primary_caps_viz'.replace(str_replace[0],str_replace[1])
    out_dir_im = os.path.join(out_dir_meta,'im_all_patches/train')
    
    out_dir_meta = '../experiments/figures/primary_caps_viz_clusters'.replace(str_replace[0],str_replace[1])
    util.mkdir(out_dir_meta)

    caps, test_file, convnet, imsize, routes  = get_caps_compiled(routed= True)
    mags = np.linalg.norm(caps,axis = 4)
    mags_org = mags
    # print 'mags_org.shape',mags_org.shape
    mags = np.transpose(mags,(0,2,3,1))
    # print mags.shape
    mags = np.reshape(mags,(mags.shape[0]*mags.shape[1]*mags.shape[2],mags.shape[3]))
    # print mags.shape

    
    # print routes.shape

    # print test_file
    gt_class = [int(line_curr.split(' ')[1]) for line_curr in util.readLinesFromFile(test_file)]

    routes_gt = routes[gt_class,range(routes.shape[1])].squeeze()
    mag_routes = np.linalg.norm(routes_gt,axis = 2)
    # np.sum(routes_gt,axis=2)
    # 
    mag_routes = np.reshape(mag_routes,(mag_routes.shape[0],32,6,6,1))
    
    # print np.min(mag_routes),np.max(mag_routes)
    # print mag_routes.shape
    # print caps.shape
    if routed:
        caps = caps*mag_routes


    caps_org = np.array(caps)

    caps = np.transpose(caps,(0,2,3,1,4))
    # print caps.shape
    caps = np.reshape(caps,(caps.shape[0]*caps.shape[1]*caps.shape[2],caps.shape[3],caps.shape[4]))
    # print caps.shape
    # print mags.shape
    idx_helper = range(caps.shape[0])
    # print len(idx_helper)
    idx_helper = np.reshape(idx_helper,(caps_org.shape[0],caps_org.shape[2],caps_org.shape[3]))
    # print idx_helper.shape


    num_to_keep = 100
    num_clusters = 32

    for filt_num in range(caps.shape[1]):
        if mag_sorted:
            out_file_html = os.path.join(out_dir_meta,str(filt_num)+'_mag_sorted.html')
        elif routed:
            out_file_html = os.path.join(out_dir_meta,str(filt_num)+'_route_weighted.html')
        else:
            out_file_html = os.path.join(out_dir_meta,str(filt_num)+'.html')
            
        im_rows = []
        caption_rows = []
        
        caps_curr = caps[:,filt_num]    
        mags_curr = mags[:,filt_num]

        k_meaner = sklearn.cluster.KMeans(n_clusters=num_clusters)
        vec_rel = sklearn.preprocessing.normalize(caps_curr,axis = 1)
        # sklearn.preprocessing.normalize(sklearn.preprocessing.normalize(caps_curr,axis=0),axis=1) #feature normalize
        # print 'vec_rel.shape',vec_rel.shape
        print vec_rel.shape
        # numpy.random.permutation(x)
        k_meaner.fit(np.random.permutation(vec_rel))
        cluster_centers = k_meaner.cluster_centers_
        print cluster_centers.shape
        cluster_belongings = k_meaner.predict(vec_rel)
        # print cluster_centers,cluster_centers.shape

        for idx_cluster_center,cluster_center in enumerate(cluster_centers):
            if mag_sorted:
                idx_rel = np.where(cluster_belongings == idx_cluster_center)[0]
                # print idx_rel.shape
                # print idx_rel[:10]
                mag_rel = mags_curr[idx_rel]
                idx_sort = np.argsort(mag_rel)[::-1]
                idx_sort = list(idx_rel[idx_sort])
                # print idx_sort[:10]
                # raw_input()
            else:            
                cluster_center = cluster_center[np.newaxis,:]
                # print (vec_rel-cluster_center).shape
                dist = np.linalg.norm(vec_rel-cluster_center,axis = 1)
                # print dist.shape
                # print mags.shape
                # raw_input()
                idx_sort = list(np.argsort(dist))

            idx_sort = idx_sort[:num_to_keep]+idx_sort[-num_to_keep:]

            im_row = []
            caption_row =[]

            for idx_idx, idx_curr in enumerate(idx_sort):
                arg_multi_dim = np.where(idx_helper==idx_curr)
                arg_multi_dim = [arr[0] for arr in arg_multi_dim]
                
                file_curr = os.path.join(out_dir_im,'_'.join([str(val) for val in arg_multi_dim])+'.jpg')
                assert os.path.exists(file_curr)
                im_row.append(util.getRelPath(file_curr,dir_server))
                caption_row.append('%d %.4f' %(idx_idx,mags_curr[idx_curr]))
                    # str(idx_idx)+' '+str(filt_num))
                
            im_rows.append(im_row)
            caption_rows.append(caption_row)

        visualize.writeHTML(out_file_html,im_rows,caption_rows,40,40)
        print out_file_html.replace(dir_server,click_str)
        # break

    #         print cluster_center.shape

    #         raw_input()

            
    #     im_row = []
    #     caption_row =[]
    
    #     print caps_curr.shape
    #     raw_input()
    #     print np.min(mag_curr), np.max(mag_curr)
    #     idx_sort = list(np.argsort(mag_curr)[::-1])
    #     idx_sort = idx_sort[:num_to_keep]+idx_sort[-num_to_keep:]

    #     sorted_mag_curr = mag_curr[idx_sort]
    #     # print sorted_mag_curr[0],sorted_mag_curr[-1]
    #     # raw_input()
        

    #     for idx_idx, idx_curr in enumerate(idx_sort):
    #         arg_multi_dim = np.where(idx_helper==idx_curr)
    #         arg_multi_dim = [arr[0] for arr in arg_multi_dim]
    #         # print arg_multi_dim
    #         # if arg_multi_dim[1]==0 or arg_multi_dim[1]==5 or arg_multi_dim[2]==0 or arg_multi_dim[2]==5:
    #         #     continue
    #         # arg_multi_dim = [arg_multi_dim[0],max(arg_multi_dim[2],1),max(arg_multi_dim[1],1)]
    #         file_curr = os.path.join(out_dir_im,'_'.join([str(val) for val in arg_multi_dim])+'.jpg')
    #         assert os.path.exists(file_curr)
    #         im_row.append(util.getRelPath(file_curr,dir_server))
    #         caption_row.append(str(idx_idx)+' '+str(filt_num))
    #         # if len(im_row)==num_to_keep:
    #         #     break
    #     im_rows.append(im_row)
    #     caption_rows.append(caption_row)

    # visualize.writeHTML(out_file_html,im_rows,caption_rows,40,40)
    # print out_file_html.replace(dir_server,click_str)



def script_view_route_weighted_patches_sorted():
    out_dir_meta = '../experiments/figures/primary_caps_viz'.replace(str_replace[0],str_replace[1])
    out_dir_im = os.path.join(out_dir_meta,'im_all_patches/train')

    caps, test_file, convnet, imsize, routes  = get_caps_compiled(routed= True)
    
    print routes.shape

    print test_file
    gt_class = [int(line_curr.split(' ')[1]) for line_curr in util.readLinesFromFile(test_file)]

    routes_gt = routes[gt_class,range(routes.shape[1])].squeeze()
    mag_routes = np.linalg.norm(routes_gt,axis = 2)
    # np.sum(routes_gt,axis=2)
    # 
    mag_routes = np.reshape(mag_routes,(mag_routes.shape[0],32,6,6,1))
    
    print np.min(mag_routes),np.max(mag_routes)
    print mag_routes.shape
    print caps.shape
    
    caps = caps*mag_routes

    mags = np.linalg.norm(caps,axis = 4)
    # mags = mags*mag_routes

    mags_org = mags
    print 'mags_org.shape',mags_org.shape

    
    mags = np.transpose(mags,(0,2,3,1))
    print mags.shape
    mags = np.reshape(mags,(mags.shape[0]*mags.shape[1]*mags.shape[2],mags.shape[3]))
    print mags.shape
    idx_helper = range(mags.shape[0])
    print len(idx_helper)
    idx_helper = np.reshape(idx_helper,(caps.shape[0],caps.shape[2],caps.shape[3]))
    print idx_helper.shape


    num_to_keep = 100
    print 'mags_org.shape',mags_org.shape


    out_file_html = os.path.join(out_dir_meta,'mag_sorted_route_weighted.html')

    im_rows = []
    caption_rows = []

    for filt_num in range(mags.shape[1]):
        im_row = []
        caption_row =[]
        mag_curr = mags[:,filt_num]
        print np.min(mag_curr), np.max(mag_curr)
        idx_sort = list(np.argsort(mag_curr)[::-1])
        idx_sort = idx_sort[:num_to_keep]+idx_sort[-num_to_keep:]

        sorted_mag_curr = mag_curr[idx_sort]
        # print sorted_mag_curr[0],sorted_mag_curr[-1]
        # raw_input()
        

        for idx_idx, idx_curr in enumerate(idx_sort):
            arg_multi_dim = np.where(idx_helper==idx_curr)
            arg_multi_dim = [arr[0] for arr in arg_multi_dim]
            # print arg_multi_dim
            # if arg_multi_dim[1]==0 or arg_multi_dim[1]==5 or arg_multi_dim[2]==0 or arg_multi_dim[2]==5:
            #     continue
            # arg_multi_dim = [arg_multi_dim[0],max(arg_multi_dim[2],1),max(arg_multi_dim[1],1)]
            file_curr = os.path.join(out_dir_im,'_'.join([str(val) for val in arg_multi_dim])+'.jpg')
            assert os.path.exists(file_curr)
            im_row.append(util.getRelPath(file_curr,dir_server))
            caption_row.append(str(idx_idx)+' '+str(filt_num))
            # if len(im_row)==num_to_keep:
            #     break
        im_rows.append(im_row)
        caption_rows.append(caption_row)

    visualize.writeHTML(out_file_html,im_rows,caption_rows,40,40)
    print out_file_html.replace(dir_server,click_str)


def make_primary_caps_emotion_map():
    out_dir_meta = '../experiments/figures'.replace(str_replace[0],str_replace[1])
    out_dir_im = os.path.join(out_dir_meta,'primary_caps_emotion_gridding')
    util.mkdir(out_dir_im)

    caps, test_file, convnet, imsize  = get_caps_compiled(routed= False)
    gt_class = np.array([int(line_curr.split(' ')[1]) for line_curr in util.readLinesFromFile(test_file)])
    
    num_emotions = np.unique(gt_class).size
    print 'num_emotions',num_emotions

    mags = np.linalg.norm(caps,axis = 4)
    mags_org = mags
    mags = np.transpose(mags,(0,2,3,1))
    
    out_file_html = os.path.join(out_dir_im,'emotion_grid.html')
    im_rows = []
    caption_rows = []
    for filt_num in range(mags.shape[-1]):
        mags_rel = mags[:,:,:,filt_num]
        im_row = []
        caption_row = []
        for emo_num in np.unique(gt_class):
            mags_emo_rel = mags_rel[gt_class==emo_num,:,:]
            mags_emo_rel = np.mean(mags_emo_rel,0)
            min_val = np.min(mags_emo_rel)
            max_val = np.max(mags_emo_rel)
            print mags_emo_rel.shape,min_val,max_val

            title_curr = '_'.join([str(val) for val in [filt_num,emo_num]])
            out_file_curr = os.path.join(out_dir_im,title_curr+'.jpg')
            visualize.plot_colored_mats(out_file_curr,mags_emo_rel,min_val,max_val, title=title_curr)
            im_row.append(util.getRelPath(out_file_curr,dir_server))
            caption_row.append(title_curr)
        im_rows.append(im_row)
        caption_rows.append(caption_row)

    visualize.writeHTML(out_file_html,im_rows,caption_rows,50,50)


def get_primary_caps_emotion_dot():
    out_dir_meta = '../experiments/figures'.replace(str_replace[0],str_replace[1])
    out_dir_im = os.path.join(out_dir_meta,'primary_caps_emotion_dotting')
    util.mkdir(out_dir_im)

    caps, test_file, convnet, imsize  = get_caps_compiled(routed= False)
    gt_class = np.array([int(line_curr.split(' ')[1]) for line_curr in util.readLinesFromFile(test_file)])
    
    num_emotions = np.unique(gt_class).size
    print 'num_emotions',num_emotions

    mags = np.linalg.norm(caps,axis = 4)
    mags_org = mags
    mags = np.transpose(mags,(0,2,3,1))
    
    # out_file_html = os.path.join(out_dir_im,'emotion_grid.html')
    # im_rows = []
    # caption_rows = []

    emos = range(8)



    all_vecs = []
    for emo_num in emos:
        emo_vec = []

        for filt_num in range(mags.shape[-1]):
            mags_rel = mags[:,:,:,filt_num]

            # im_row = []
            # caption_row = []
            # vec_filt = []
            mags_emo_rel = mags_rel[gt_class==emo_num,:,:]
            mags_emo_rel = np.mean(mags_emo_rel,0)
            # mags_emo_rel = mags_emo_rel[1:5,1:5]
            mags_emo_rel = mags_emo_rel.flatten()
            emo_vec.append(mags_emo_rel)

        # vec_filt.append(mags_emo_rel)
        # emo_vec = np.concatenate(vec_filt,0)
        # print vec_filt.shape
        all_vecs.append(emo_vec)

    for emo_num in emos:
        vecs = all_vecs[emo_num]
        vecs = np.array(vecs)
        vecs = vecs/np.linalg.norm(vecs,axis=1,keepdims=True)
        dot_prods = np.zeros((vecs.shape[0],vecs.shape[0]))
        for r in range(vecs.shape[0]):
            for c in range(vecs.shape[0]):
                dotter = np.dot(vecs[r,:],vecs[c,:])
                dot_prods[r,c]=dotter

        out_file_curr = os.path.join(out_dir_im,str(emo_num)+'.jpg')
        visualize.plot_colored_mats(out_file_curr,dot_prods,np.min(dot_prods),np.max(dot_prods), title=str(emo_num))
        # print vecs.shape
        # raw_input()

    all_vecs = [np.concatenate([all_vecs[val][filt] for val in emos],0) for filt in range(mags.shape[-1])]
    print len(all_vecs)
    print all_vecs[0].shape

    all_vecs = np.array(all_vecs)
    print all_vecs.shape

    all_vecs = all_vecs/np.linalg.norm(all_vecs,axis=1,keepdims=True)
    print all_vecs.shape
    print np.linalg.norm(all_vecs[0])

    dot_prods = np.zeros((all_vecs.shape[0],all_vecs.shape[0]))
    for r in range(all_vecs.shape[0]):
        for c in range(all_vecs.shape[0]):
            dotter = np.dot(all_vecs[r,:],all_vecs[c,:])
            dot_prods[r,c]=dotter

    out_file_curr = os.path.join(out_dir_im,'all.jpg')
    visualize.plot_colored_mats(out_file_curr,dot_prods,np.min(dot_prods),np.max(dot_prods), title='all')

    visualize.writeHTMLForFolder(out_dir_im)

    
    # visualize.plot_colored_mats(out_file_curr,dot_prods,np.min(dot_prods),np.max(dot_prods),title='Dot Prods All emo vec')
    # print out_file_curr.replace(dir_server,click_str)
    
            # min_val = np.min(mags_emo_rel)
            # max_val = np.max(mags_emo_rel)
            # print mags_emo_rel.shape,min_val,max_val
            # raw_input()
    #         title_curr = '_'.join([str(val) for val in [filt_num,emo_num]])
    #         out_file_curr = os.path.join(out_dir_im,title_curr+'.jpg')
    #         visualize.plot_colored_mats(out_file_curr,mags_emo_rel,min_val,max_val, title=title_curr)
    #         im_row.append(util.getRelPath(out_file_curr,dir_server))
    #         caption_row.append(title_curr)
    #     im_rows.append(im_row)
    #     caption_rows.append(caption_row)

    # visualize.writeHTML(out_file_html,im_rows,caption_rows,50,50)



def change_direction_and_retrieve():
    out_dir_meta = '../experiments/figures/primary_caps_viz'.replace(str_replace[0],str_replace[1])
    out_dir_im = os.path.join(out_dir_meta,'im_all_patches/train')
    
    out_dir_meta = '../experiments/figures/primary_caps_viz_change_direction'.replace(str_replace[0],str_replace[1])
    util.mkdir(out_dir_meta)

    caps, test_file, convnet, imsize, routes  = get_caps_compiled(routed= True)
    mags = np.linalg.norm(caps,axis = 4)
    mags = np.transpose(mags,(0,2,3,1))
    # mags = np.reshape(mags,(mags.shape[0]*mags.shape[1]*mags.shape[2],mags.shape[3]))
    
    gt_class = [int(line_curr.split(' ')[1]) for line_curr in util.readLinesFromFile(test_file)]

    caps_org = np.array(caps)

    caps = np.transpose(caps,(0,2,3,1,4))
    # caps = np.reshape(caps,(caps.shape[0]*caps.shape[1]*caps.shape[2],caps.shape[3],caps.shape[4]))
    idx_helper = range(caps.shape[0]*caps.shape[1]*caps.shape[2])
    idx_helper = np.reshape(idx_helper,(caps_org.shape[0],caps_org.shape[2],caps_org.shape[3]))

    print caps.shape
    print mags.shape
    raw_input()

    num_to_keep = 100
    num_clusters = 32


    # range_values = np.arange(-0.5,
    mag_range = np.arange(-0.5,0.6,0.05)

    for filt_num in range(caps.shape[3]):
        out_file_html = os.path.join(out_dir_meta,'_'.join([str(val) for val in [filt_num]])+'.html')
        html_rows = []
        html_captions = []

        for row_num in range(1,6):
            for col_num in range(1,6):

                
                caps_rel = caps[:,row_num,col_num,filt_num,:]
                mags_rel = mags[:,row_num,col_num,filt_num]

                idx_max = np.argmax(mags_rel,0);
                caps_max = caps_rel[idx_max];

                caps_norms = np.linalg.norm(caps_rel,axis=1, keepdims=True)
                caps_unit = caps_rel

                print caps_norms.shape
                print caps_rel.shape

                rel_idx = [idx_max,row_num,col_num]
                file_max = os.path.join(out_dir_im,'_'.join([str(val) for val in rel_idx])+'.jpg')
                
                for caps_dim in range(caps_max.shape[0]):

                    caps_max = caps_max/np.linalg.norm(caps_max)

                    html_row_curr = [file_max]
                    caption_row_curr = ['org %d %d %.2f' % (filt_num,caps_dim,mags_rel[idx_max])]
                    for dim_mag_curr in mag_range:

                        caps_new = caps_max[:]
                        caps_new[caps_dim] = dim_mag_curr
                        # print caps_new
                        caps_new = caps_new/np.linalg.norm(caps_new,keepdims = True)
                        # print caps_new
                        distances = np.abs(np.matmul(caps_unit,caps_new[:,np.newaxis]))
                        # min_idx = np.argsort(distances);
                        closest_idx = np.argmin(distances)
                        rel_idx = [closest_idx,row_num,col_num]

                        file_curr = os.path.join(out_dir_im,'_'.join([str(val) for val in rel_idx])+'.jpg')
                        caption_curr = '%.2f' % (dim_mag_curr)
                        # , distances[closest_idx]) 
                        html_row_curr.append(file_curr);
                        caption_row_curr.append(caption_curr);

                    html_row_curr = [util.getRelPath(file_curr,dir_server) for file_curr in html_row_curr]

                    html_rows.append(html_row_curr)
                    html_captions.append(caption_row_curr)

        visualize.writeHTML(out_file_html,html_rows,html_captions,40,40)
        print out_file_html


                        # arg_multi_dim = np.where(idx_helper==idx_curr)
        #         arg_multi_dim = [arr[0] for arr in arg_multi_dim]
                
        #         file_curr = os.path.join(out_dir_im,'_'.join([str(val) for val in arg_multi_dim])+'.jpg')
        #         assert os.path.exists(file_curr)

                        # print distances.shape,closest_idx,distances[closest_idx]
                        # print closest_idx
                        # print closest_idx.shape
                        # print np.min(closest_idx),np.max(closest_idx)

                        # raw_input()
                        



                # print mags_rel[idx_max]
                # print caps_rel[idx_max]

                # print caps_rel.shape
                # print mags_rel.shape
                # raw_input()



        # if mag_sorted:
        #     out_file_html = os.path.join(out_dir_meta,str(filt_num)+'_mag_sorted.html')
        # elif routed:
        #     out_file_html = os.path.join(out_dir_meta,str(filt_num)+'_route_weighted.html')
        # else:
        #     out_file_html = os.path.join(out_dir_meta,str(filt_num)+'.html')
            
        # im_rows = []
        # caption_rows = []
        
        # caps_curr = caps[:,filt_num]    
        # mags_curr = mags[:,filt_num]

        # k_meaner = sklearn.cluster.KMeans(n_clusters=num_clusters)
        # vec_rel = sklearn.preprocessing.normalize(caps_curr,axis = 1)
        # # sklearn.preprocessing.normalize(sklearn.preprocessing.normalize(caps_curr,axis=0),axis=1) #feature normalize
        # # print 'vec_rel.shape',vec_rel.shape
        # print vec_rel.shape
        # # numpy.random.permutation(x)
        # k_meaner.fit(np.random.permutation(vec_rel))
        # cluster_centers = k_meaner.cluster_centers_
        # print cluster_centers.shape
        # cluster_belongings = k_meaner.predict(vec_rel)
        # # print cluster_centers,cluster_centers.shape

        # for idx_cluster_center,cluster_center in enumerate(cluster_centers):
        #     if mag_sorted:
        #         idx_rel = np.where(cluster_belongings == idx_cluster_center)[0]
        #         # print idx_rel.shape
        #         # print idx_rel[:10]
        #         mag_rel = mags_curr[idx_rel]
        #         idx_sort = np.argsort(mag_rel)[::-1]
        #         idx_sort = list(idx_rel[idx_sort])
        #         # print idx_sort[:10]
        #         # raw_input()
        #     else:            
        #         cluster_center = cluster_center[np.newaxis,:]
        #         # print (vec_rel-cluster_center).shape
        #         dist = np.linalg.norm(vec_rel-cluster_center,axis = 1)
        #         # print dist.shape
        #         # print mags.shape
        #         # raw_input()
        #         idx_sort = list(np.argsort(dist))

        #     idx_sort = idx_sort[:num_to_keep]+idx_sort[-num_to_keep:]

        #     im_row = []
        #     caption_row =[]

        #     for idx_idx, idx_curr in enumerate(idx_sort):
        #         arg_multi_dim = np.where(idx_helper==idx_curr)
        #         arg_multi_dim = [arr[0] for arr in arg_multi_dim]
                
        #         file_curr = os.path.join(out_dir_im,'_'.join([str(val) for val in arg_multi_dim])+'.jpg')
        #         assert os.path.exists(file_curr)
        #         im_row.append(util.getRelPath(file_curr,dir_server))
        #         caption_row.append('%d %.4f' %(idx_idx,mags_curr[idx_curr]))
        #             # str(idx_idx)+' '+str(filt_num))
                
        #     im_rows.append(im_row)
        #     caption_rows.append(caption_row)

        # visualize.writeHTML(out_file_html,im_rows,caption_rows,40,40)
        # print out_file_html.replace(dir_server,click_str)
 



def main():
    change_direction_and_retrieve()
    # get_primary_caps_emotion_dot()
    # make_primary_caps_emotion_map()

    # script_view_clusters_high_mag(mag_sorted = False, mag_percent = 0.5)
    # script_view_clusters_high_mag(mag_sorted = True, mag_percent = 0.5)


    # script_view_clusters(routed = False,mag_sorted = True)
    # script_view_all_patches_sorted()
    # script_view_route_weighted_clusters()
    # script_view_route_weighted_clusters()

    
    # script_viz_mag()
    # script_viz_k_means()





    # for x in range(6):
    #     for y in range(6):
    #         rec_field, center = receptive_field.get_receptive_field(convnet,imsize,4, x,y)
    #         center = [int(round(val)) for val in center]
    #         range_x = [max(0,center[0]-rec_field/2),min(imsize,center[0]+rec_field/2)]
    #         range_y = [max(0,center[1]-rec_field/2),min(imsize,center[1]+rec_field/2)]
    #         print x, y , range_x, range_y


if __name__=='__main__':
    main()