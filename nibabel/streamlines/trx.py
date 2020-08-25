import json
import mmap
import os
import shutil
import tempfile
import zipfile

from dipy.io.stateful_tractogram import StatefulTractogram, Space
from dipy.io.utils import get_reference_info
from nibabel.affines import voxel_sizes
from nibabel.orientations import aff2axcodes
from nibabel.streamlines.array_sequence import ArraySequence

import numpy as np


def _create_memmap(filename, mode='r', shape=(1,), dtype=np.float32, offset=0):
    if shape[0]:
        return np.memmap(filename, mode=mode, offset=offset,
                         shape=shape,  dtype=dtype)
    else:
        if not os.path.isfile(filename):
            f = open(filename, "wb")
            f.close
        return np.zeros(shape, dtype=dtype)


def load(input_obj):
    # Check if 0 streamlines, if yes then 0 points is expected (vice-versa)
    # 4x4 affine matrices should contains values (no all-zeros)
    # 3x1 dimensions array should contains values at each position (int)
    # Catch the error if filename do not have a dtype extension (support bool?)
    if os.path.isfile(input_obj):
        was_compressed = False
        with zipfile.ZipFile(input_obj, 'r') as zf:
            for info in zf.infolist():
                if info.compress_type != 0:
                    was_compressed = True
                    break
        if was_compressed:
            with zipfile.ZipFile(input_obj, 'r') as zf:
                tmpdir = tempfile.TemporaryDirectory()
                zf.extractall(tmpdir.name)
                trx = load_from_directory(tmpdir.name)
                trx._uncompressed_folder_handle = tmpdir
                print('File was compressed, call the close() function before quitting!')
        else:
            trx = load_from_zip(input_obj)
    elif os.path.isdir(input_obj):
        trx = load_from_directory(input_obj)
    else:
        raise ValueError('File/Folder does not exist')

    # Example of robust check for metadata
    for dpg in trx.data_per_group.keys():
        if dpg not in trx.groups.keys():
            raise ValueError('An undeclared group ({}) has data_per_group.'.format(
                dpg))

    return trx


def concatenate(trx_list, delete_dpp=False, delete_dps=False, delete_dpg=False,
                delete_groups=False, use_ref_dtype=True,
                check_space_attributes=True, preallocation_count=None):
    ref_trx = trx_list[0]

    if check_space_attributes:
        for curr_trx in trx_list[1:]:
            if not np.allclose(ref_trx.header['affine'], curr_trx.header['affine']) \
                    or not np.array_equal(ref_trx.header['dimensions'], curr_trx.header['dimensions']):
                raise ValueError('Wrong space attributes')

    if preallocation_count is not None and not (delete_groups or delete_dpg):
        raise ValueError('Groups are variables, cannot be handled with preallocation')

    if delete_dpp:
        ref_trx.data_per_point = {}
    if delete_dps:
        ref_trx.data_per_streamline = {}
    if delete_dpg:
        ref_trx.data_per_group = {}
    for curr_trx in trx_list[1:]:
        for key in curr_trx.data_per_point.keys():
            if key not in ref_trx.data_per_point.keys():
                if delete_dpp:
                    print('No same dpp, deleting')
                    curr_trx.data_per_point = {}
                else:
                    raise ValueError('Not same dpp keys')
            elif not use_ref_dtype \
                and (ref_trx.data_per_point[key].get_data().dtype
                     != curr.data_per_point[key].get_data().dtype):
                raise ValueError('Not same dpp dtype')
    for curr_trx in trx_list[1:]:
        for key in curr_trx.data_per_streamline.keys():
            if key not in ref_trx.data_per_streamline.keys():
                if delete_dps:
                    print('No same dpp, deleting')
                    curr_trx.data_per_point = {}
                else:
                    raise ValueError('Not same dpp keys')
            elif not use_ref_dtype \
                and (ref_trx.data_per_streamline[key].dtype
                     != curr.data_per_streamline[key].dtype):
                raise ValueError('Not same dpp dtype')

    all_groups_len = {}
    all_groups_dtype = {}
    if delete_dpg or delete_groups:
        for cur_trx in trx_list:
            if delete_dpg:
                cur_trx.data_per_group = {}
            if delete_groups:
                cur_trx.groups = {}

    if not (delete_dpg and delete_groups):
        for trx_1 in trx_list:
            for key in trx_1.groups.keys():
                if key in all_groups_len:
                    all_groups_len[key] += len(trx_1.groups[key])
                else:
                    all_groups_len[key] = len(trx_1.groups[key])
                    all_groups_dtype[key] = trx_1.groups[key].dtype
                for trx_2 in trx_list:
                    if key in trx_2.groups.keys():
                        if key in trx_1.data_per_group:
                            for sub_key in trx_1.data_per_group[key].keys():
                                if sub_key in trx_2.data_per_group[key].keys():
                                    raise ValueError('Same dpg keys, cant fuse, use delete_dpg')
                        if not use_ref_dtype \
                                and trx_1.groups[key].dtype != trx_2.groups[key].dtype:
                            raise ValueError('Not same group dtype')

    nbr_points = 0
    nbr_streamlines = 0
    to_concat_list = trx_list[1:] if preallocation_count is not None else trx_list

    for curr_trx in to_concat_list:
        nbr_points += len(curr_trx.streamlines.get_data())
        nbr_streamlines += len(curr_trx.streamlines)

    if preallocation_count is None or ref_trx.header['nbr_points'] < nbr_points \
            or ref_trx.header['nbr_streamlines'] < nbr_streamlines:
        preallocation_count = 0
        points_preallocation_count = 0
        to_concat_list = trx_list
        new_trx = TrxFile(nbr_points=nbr_points, nbr_streamlines=nbr_streamlines,
                          init_as=ref_trx)
        for group_key in all_groups_len.keys():
            dtype = all_groups_dtype[group_key]
            group_filename = os.path.join(new_trx._uncompressed_folder_handle.name, 'groups'
                                          '{}.{}'.format(group_key, dtype.name))
            new_trx.groups[group_key] = _create_memmap(group_filename, mode='w+',
                                                       shape=(all_groups_len[group_key],),
                                                       dtype=dtype)
            pos = 0
            count = 0
            for curr_trx in trx_list:
                new_trx.groups[group_key][pos:pos +
                                          len(curr_trx.groups[group_key])] = curr_trx.groups[group_key] + count
                pos += len(curr_trx.groups[group_key])
                count += curr_trx.header['nbr_streamlines']

        for curr_trx in trx_list:
            for group_key in curr_trx.data_per_group.keys():
                for dpg_key in curr_trx.data_per_group[group_key].keys():
                    if group_key not in new_trx.data_per_group:
                        new_trx.data_per_group[group_key] = {}
                    dtype = curr_trx.data_per_group[group_key][dpg_key].dtype
                    dpg_filename = os.path.join(tmp_dir, 'dpg', group_key,
                                                '{}.{}'.format(dpg_key, dtype.name))
                    new_trx.data_per_group[group_key][dpg_key] = _create_memmap(dpg_filename, mode='w+',
                                                                                shape=(1,),
                                                                                dtype=dtype)
                    new_trx.data_per_group[group_key][dpg_key][:
                                                               ] = curr_trx.data_per_group[group_key][dpg_key]
    else:
        new_trx = ref_trx
        if not np.any(new_trx.streamlines._offsets):
            preallocation_count = 0
            points_preallocation_count = 0
        else:
            preallocation_count = int(np.nonzero(new_trx.streamlines._offsets)[0][-1]+1)
            points_preallocation_count = int(np.sum(new_trx.streamlines._lengths[0:preallocation_count]))

    for curr_trx in to_concat_list:
        new_trx.streamlines._data[points_preallocation_count:points_preallocation_count +
                                  len(curr_trx.streamlines.get_data())] = curr_trx.streamlines.get_data()
        new_trx.streamlines._offsets[preallocation_count:preallocation_count+len(
            curr_trx.streamlines._offsets)] = curr_trx.streamlines._offsets + points_preallocation_count
        new_trx.streamlines._lengths[preallocation_count:preallocation_count +
                                     len(curr_trx.streamlines._lengths)] = curr_trx.streamlines._lengths
        for dpp_key in new_trx.data_per_point.keys():
            new_trx.data_per_point[dpp_key]._data[points_preallocation_count:points_preallocation_count +
                                                  len(curr_trx.streamlines.get_data())] = curr_trx.data_per_point[dpp_key].get_data()
            new_trx.data_per_point[dpp_key]._offsets[preallocation_count:preallocation_count+len(
                curr_trx.streamlines._offsets)] = curr_trx.streamlines._offsets + points_preallocation_count
            new_trx.data_per_point[dpp_key]._lengths[preallocation_count:preallocation_count +
                                                     len(curr_trx.streamlines._lengths)] = curr_trx.streamlines._lengths
        for dps_key in new_trx.data_per_streamline.keys():
            new_trx.data_per_streamline[dps_key][preallocation_count:preallocation_count +
                                                 len(curr_trx.streamlines)] = curr_trx.data_per_streamline[dps_key]
        preallocation_count += len(curr_trx.streamlines._lengths)
        points_preallocation_count += int(np.sum(curr_trx.streamlines._lengths))

    return new_trx


def load_from_zip(filename):
    trx = TrxFile()
    with zipfile.ZipFile(filename, mode='r') as zf:
        with zf.open('header.json') as zf_header:
            data = zf_header.read()
            trx.header = json.loads(data)
            trx.nbr_points = trx.header['nbr_points']
            trx.nbr_streamlines = trx.header['nbr_streamlines']

        files_pointer_size = {}
        for zip_info in zf.filelist:
            elem_filename = zip_info.filename
            if elem_filename == 'header.json':
                continue
            _, ext = os.path.splitext(elem_filename)

            mem_adress = zip_info.header_offset + len(zip_info.FileHeader())
            size = zip_info.file_size / np.dtype(ext[1:]).itemsize

            if size.is_integer():
                files_pointer_size[elem_filename] = mem_adress, int(size)
            else:
                raise ValueError('Wrong size or datatype')
    return create_trx_from_memmap(trx, files_pointer_size, root_zip=filename)


def load_from_directory(directory):
    directory = os.path.abspath(directory)
    trx = TrxFile()
    with open(os.path.join(directory, 'header.json')) as header:
        trx.header = json.load(header)
        trx.nbr_points = trx.header['nbr_points']
        trx.nbr_streamlines = trx.header['nbr_streamlines']

    files_pointer_size = {}
    for root, dirs, files in os.walk(directory):
        for name in files:
            elem_filename = os.path.join(root, name)
            if name == 'header.json':
                continue
            _, ext = os.path.splitext(elem_filename)

            size = os.path.getsize(elem_filename) / np.dtype(ext[1:]).itemsize
            if size.is_integer():
                files_pointer_size[elem_filename] = 0, int(size)
            elif os.path.getsize(elem_filename) == 1:
                files_pointer_size[elem_filename] = 0, 0
            else:
                raise ValueError('Wrong size or datatype')

    return create_trx_from_memmap(trx, files_pointer_size, root=directory)


def create_trx_from_memmap(trx, dict_pointer_size, root_zip=None, root=None):
    for elem_filename in dict_pointer_size.keys():
        if root_zip:
            filename = root_zip
        else:
            filename = elem_filename
        base, ext = os.path.splitext(elem_filename)
        folder = os.path.dirname(base)
        base = os.path.basename(base)
        mem_adress, size = dict_pointer_size[elem_filename]

        if root is not None and folder.startswith(root.rstrip('/')):
            folder = folder.replace(root, '').lstrip('/')

        if base in ['data', 'offsets', 'lengths'] and folder == '':
            if base == 'data':
                if size != trx.nbr_points*3:
                    raise ValueError('Wrong data size')
                data = _create_memmap(filename, mode='r+',
                                      offset=mem_adress,
                                      shape=(trx.nbr_points, 3),
                                      dtype=ext[1:])
            elif base == 'offsets':
                if size != trx.nbr_streamlines:
                    raise ValueError('Wrong offsets size')
                offsets = _create_memmap(filename, mode='r+',
                                         offset=mem_adress,
                                         shape=(trx.nbr_streamlines,),
                                         dtype=ext[1:])
            elif base == 'lengths':
                if size != trx.nbr_streamlines:
                    raise ValueError('Wrong lengths size')
                lengths = _create_memmap(filename, mode='r+',
                                         offset=mem_adress,
                                         shape=(trx.nbr_streamlines,),
                                         dtype=ext[1:])
        else:
            if folder == 'dps':
                if size != trx.nbr_streamlines:
                    raise ValueError('Wrong dps size')
                trx.data_per_streamline[base] = _create_memmap(
                    filename, mode='r+', offset=mem_adress,
                    shape=(trx.nbr_streamlines,), dtype=ext[1:])

            elif folder == 'dpp':
                if size != trx.nbr_points:
                    raise ValueError('Wrong dpp size')
                trx.data_per_point[base] = _create_memmap(
                    filename, mode='r+', offset=mem_adress,
                    shape=(trx.nbr_points, 1), dtype=ext[1:])
            elif folder.startswith('dpg'):
                if size != 1:
                    raise ValueError('Wrong lengths size')
                data_name = os.path.basename(base)
                sub_folder = os.path.basename(folder)
                if sub_folder not in trx.data_per_group:
                    trx.data_per_group[sub_folder] = {}
                trx.data_per_group[sub_folder][data_name] = _create_memmap(
                    filename, mode='r+', offset=mem_adress,
                    shape=(1,), dtype=ext[1:])
            elif folder == 'groups':
                trx.groups[base] = _create_memmap(
                    filename, mode='r+', offset=mem_adress,
                    shape=(size,), dtype=ext[1:])

    if data is not None \
        and offsets is not None \
            and lengths is not None:
        trx.streamlines = ArraySequence()
        trx.streamlines._data = data
        trx.streamlines._offsets = offsets
        trx.streamlines._lengths = lengths
    else:
        raise ValueError('Missing essential data')

    for dpp_key in trx.data_per_point:
        tmp = trx.data_per_point[dpp_key]
        trx.data_per_point[dpp_key] = ArraySequence()
        trx.data_per_point[dpp_key]._data = tmp
        trx.data_per_point[dpp_key]._offsets = offsets
        trx.data_per_point[dpp_key]._lengths = lengths

    return trx


def save(trx, filename):
    trx.resize()
    if os.path.isfile(filename):
        raise ValueError('file already exists, use Concatenate()')
    with tempfile.TemporaryDirectory() as tmp_dir:
        with open(os.path.join(tmp_dir, 'header.json'), 'w') as out_json:
            json.dump(trx.header, out_json)

        trx.streamlines._data.tofile(os.path.join(tmp_dir, 'data.{}'.format(
            trx.streamlines._data.dtype.name)))
        trx.streamlines._offsets.tofile(os.path.join(tmp_dir, 'offsets.{}'.format(
            trx.streamlines._offsets.dtype.name)))
        trx.streamlines._lengths.tofile(os.path.join(tmp_dir, 'lengths.{}'.format(
            trx.streamlines._lengths.dtype.name)))

        if len(trx.data_per_point.keys()) > 0:
            os.mkdir(os.path.join(tmp_dir, 'dpp/'))
        for dpp_key in trx.data_per_point.keys():
            to_dump = trx.data_per_point[dpp_key].get_data()
            to_dump.tofile(os.path.join(tmp_dir,
                                        'dpp/{}.{}'.format(dpp_key,
                                                           to_dump.dtype.name)))
        if len(trx.data_per_streamline.keys()) > 0:
            os.mkdir(os.path.join(tmp_dir, 'dps/'))
        for dps_key in trx.data_per_streamline.keys():
            to_dump = trx.data_per_streamline[dps_key]
            to_dump.tofile(os.path.join(tmp_dir,
                                        'dps/{}.{}'.format(dps_key,
                                                           to_dump.dtype.name)))
        if len(trx.data_per_group.keys()) > 0:
            os.mkdir(os.path.join(tmp_dir, 'dpg/'))
        for group_key in trx.data_per_group.keys():
            for dpg_key in trx.data_per_group[group_key].keys():
                os.mkdir(os.path.join(tmp_dir, 'dpg/', group_key))
                to_dump = trx.data_per_group[group_key][dpg_key]
                to_dump.tofile(os.path.join(tmp_dir,
                                            'dpg/{}/{}.{}'.format(group_key,
                                                                  dpg_key,
                                                                  to_dump.dtype.name)))

        if len(trx.groups.keys()) > 0:
            os.mkdir(os.path.join(tmp_dir, 'groups/'))
        for group_key in trx.groups.keys():
            to_dump = trx.groups[group_key]
            to_dump.tofile(os.path.join(tmp_dir,
                                        'groups/{}.{}'.format(group_key,
                                                              to_dump.dtype.name)))
        if os.path.splitext(filename)[1]:
            zip_from_folder(tmp_dir, filename)
        else:
            if os.path.isdir(filename):
                shutil.rmtree(filename)
            shutil.copytree(tmp_dir, filename)


def zip_from_folder(directory, filename):
    with zipfile.ZipFile(filename, mode='w') as zf:
        for root, dirs, files in os.walk(directory):
            for name in files:
                tmp_filename = os.path.join(root, name)
                zf.write(tmp_filename, tmp_filename.replace(directory+'/', ''))


class TrxFile():
    def __init__(self, nbr_points=None, nbr_streamlines=None, init_as=None,
                 reference=None):

        if nbr_points is None and nbr_streamlines is None:
            if init_as is not None:
                raise ValueError('Cant use init_as without declaring nbr_')
            self.header = {}
            self.streamlines = ArraySequence()
            self.groups = {}
            self.data_per_streamline = {}
            self.data_per_point = {}
            self.data_per_group = {}
            self._uncompressed_folder_handle = None

            return

        if nbr_points is not None and nbr_streamlines is not None:
            if init_as is not None:
                affine = init_as.header['affine']
                dimensions = init_as.header['dimensions']
            elif reference is not None:
                affine, dimensions, _, _ = get_reference_info(reference)
            else:
                raise ValueError('You must declare a reference if not providing another trx')

            trx = self._initialize_empty_trx(nbr_streamlines, nbr_points,
                                             init_as=init_as)
            trx.header['affine'] = affine
            trx.header['dimensions'] = dimensions
            trx.header['nbr_points'] = nbr_points
            trx.header['nbr_streamlines'] = nbr_streamlines

            self.__dict__ = trx.__dict__
            return
        else:
            raise ValueError('You must declare both nbr_')

    @staticmethod
    def _initialize_empty_trx(nbr_streamlines, nbr_points, init_as=None):
        trx = TrxFile()

        if init_as is not None:
            data_dtype = init_as.streamlines.get_data().dtype
            offsets_dtype = init_as.streamlines._offsets.dtype
            lengths_dtype = init_as.streamlines._lengths.dtype
        else:
            data_dtype = np.dtype(np.float16)
            offsets_dtype = np.dtype(np.uint64)
            lengths_dtype = np.dtype(np.uint32)

        tmp_dir = tempfile.TemporaryDirectory()
        data_filename = os.path.join(tmp_dir.name,
                                     'data.{}'.format(data_dtype.name))
        trx.streamlines._data = _create_memmap(data_filename, mode='w+',
                                               shape=(nbr_points, 3),
                                               dtype=data_dtype)

        offsets_filename = os.path.join(tmp_dir.name,
                                        'offsets.{}'.format(offsets_dtype.name))
        trx.streamlines._offsets = _create_memmap(offsets_filename, mode='w+',
                                                  shape=(nbr_streamlines,),
                                                  dtype=offsets_dtype)

        lengths_filename = os.path.join(tmp_dir.name,
                                        'lengths.{}'.format(lengths_dtype.name))
        trx.streamlines._lengths = _create_memmap(lengths_filename, mode='w+',
                                                  shape=(nbr_streamlines,),
                                                  dtype=lengths_dtype)
        if init_as is not None:
            if len(init_as.data_per_point.keys()) > 0:
                os.mkdir(os.path.join(tmp_dir.name, 'dpp/'))
            if len(init_as.data_per_streamline.keys()) > 0:
                os.mkdir(os.path.join(tmp_dir.name, 'dps/'))
            for dpp_key in init_as.data_per_point.keys():
                dtype = init_as.data_per_point[dpp_key].get_data().dtype
                dpp_filename = os.path.join(tmp_dir.name, 'dpp/'
                                            '{}.{}'.format(dpp_key, dtype.name))
                trx.data_per_point[dpp_key] = ArraySequence()
                trx.data_per_point[dpp_key]._data = _create_memmap(dpp_filename, mode='w+',
                                                                   shape=(nbr_points, 1),
                                                                   dtype=dtype)
                trx.data_per_point[dpp_key]._offsets = trx.streamlines._offsets
                trx.data_per_point[dpp_key]._lengths = trx.streamlines._lengths

            for dps_key in init_as.data_per_streamline.keys():
                dtype = init_as.data_per_streamline[dps_key].dtype
                dps_filename = os.path.join(tmp_dir.name, 'dps/'
                                            '{}.{}'.format(dps_key, dtype.name))
                trx.data_per_streamline[dps_key] = _create_memmap(dps_filename, mode='w+',
                                                                  shape=(nbr_streamlines,),
                                                                  dtype=dtype)

        trx._uncompressed_folder_handle = tmp_dir

        return trx

    def resize(self, nbr_streamlines=None, nbr_points=None, delete_dpg=False):
        if len(self.streamlines):
            real_nbr_streamlines = int(np.nonzero(self.streamlines._offsets)[0][-1]+1)
            if nbr_streamlines is not None and nbr_streamlines < real_nbr_streamlines:
                real_nbr_streamlines = nbr_streamlines
            real_nbr_points = int(np.sum(self.streamlines._lengths[0:real_nbr_streamlines]))
        else:
            real_nbr_streamlines = 0
            real_nbr_points = 0

        if nbr_streamlines is None:
            nbr_streamlines = real_nbr_streamlines
            if nbr_streamlines == self.header['nbr_streamlines']:
                return
        
        if nbr_points is None:
            nbr_points = real_nbr_points

        trx = self._initialize_empty_trx(nbr_streamlines, nbr_points, init_as=self)
        trx.header['affine'] = self.header['affine']
        trx.header['dimensions'] = self.header['dimensions']
        trx.header['nbr_points'] = nbr_points
        trx.header['nbr_streamlines'] = nbr_streamlines

        trx.streamlines._data[0:real_nbr_points] = self.streamlines._data[0:real_nbr_points]
        trx.streamlines._offsets[0:real_nbr_streamlines] = self.streamlines._offsets[0:real_nbr_streamlines]
        trx.streamlines._lengths[0:real_nbr_streamlines] = self.streamlines._lengths[0:real_nbr_streamlines]

        for dpp_key in self.data_per_point.keys():
            trx.data_per_point[dpp_key]._data[0:real_nbr_points] = self.data_per_point[dpp_key]._data[0:real_nbr_points]
            trx.data_per_point[dpp_key]._offsets[:] = trx.streamlines._offsets
            trx.data_per_point[dpp_key]._lengths[:] = trx.streamlines._lengths

        for dps_key in self.data_per_streamline.keys():
            trx.data_per_streamline[dps_key][0:real_nbr_streamlines] = self.data_per_streamline[dps_key][0:real_nbr_streamlines]

        tmp_dir = trx._uncompressed_folder_handle
        if len(self.groups.keys()) > 0:
            os.mkdir(os.path.join(tmp_dir, 'groups/'))

        if len(trx.data_per_group.keys()) > 0:
            os.mkdir(os.path.join(tmp_dir, 'dpg/'))

        for group_key in self.groups.keys():
            group_dtype = self.groups[group_key].dtype
            group_name = os.path.join(tmp_dir, 'groups/',
                                      '{}.{}'.format(group_key,
                                                     group_dtype.name))
            tmp = self.groups[group][self.groups[group] < real_nbr_streamlines]
            trx.groups[group_key] = _create_memmap(group_name, mode='w+',
                                                   shape=(len(tmp),), dtype=group_dtype)
            trx.groups[group_key][:] = tmp

            if not delete_dpg:
                for dpg_key in self.data_per_group[group_key].keys():
                    dpg_dtype = self.data_per_group[group_key][dpg_key].dtype
                    dpg_name = os.path.join(tmp_dir, 'dpg/', group_key,
                                            '{}.{}'.format(dpg_key,
                                                           dpg_dtype.name))
                if group_key not in trx.self.data_per_group[group_key]:
                    trx.self.data_per_group[group_key] = {}
                trx.self.data_per_group[group_key][dpg_key] = _create_memmap(dpg_name, mode='w+',
                                                                             shape=(1,), dtype=dpg_dtype)

                trx.self.data_per_group[group_key][dpg_key][:
                                                            ] = self.self.data_per_group[group_key][dpg_key]

        self.close()
        self.__dict__ = trx.__dict__

    def append(self, trx, buffer_size=0):
        if len(self.streamlines):
            real_size = int(np.nonzero(self.streamlines._offsets)[0][-1]+1)
            real_point_size = int(np.sum(self.streamlines._lengths))
        else:
            real_size = 0
            real_point_size = 0

        nbr_points = real_point_size + trx.header['nbr_points']
        nbr_streamlines = real_size + trx.header['nbr_streamlines']
        if self.header['nbr_points'] < nbr_points \
                or self.header['nbr_streamlines'] < nbr_streamlines:
            self.resize(nbr_streamlines=nbr_streamlines+buffer_size,
                        nbr_points=nbr_points+buffer_size*100)
            _ = concatenate([self, trx], preallocation_count=0,
                            delete_dpg=True, delete_groups=True)

    @ staticmethod
    def from_sft(sft):
        trx = TrxFile()
        trx.header = {'dimensions': sft.dimensions.tolist(),
                      'affine': sft.affine.tolist(),
                      'nbr_points': len(sft.streamlines.get_data()),
                      'nbr_streamlines': len(sft.streamlines)}
        trx.streamlines = sft.streamlines
        trx.data_per_streamline = sft.data_per_streamline
        trx.data_per_point = sft.data_per_point

        tmpdir = tempfile.TemporaryDirectory()
        save(trx, tmpdir.name)
        trx = load_from_directory(tmpdir.name)
        trx._uncompressed_folder_handle = tmpdir

        return trx

    def to_sft(self):
        affine = np.array(self.header['affine'], dtype=np.float32)
        dimensions = np.array(self.header['dimensions'], dtype=np.uint16)
        vox_sizes = np.array(voxel_sizes(affine), dtype=np.float32)
        vox_order = ''.join(aff2axcodes(affine))
        space_attributes = (affine, dimensions, vox_sizes, vox_order)
        self.resize()
        sft = StatefulTractogram(self.streamlines, space_attributes, Space.RASMM,
                                 data_per_point=self.data_per_point,
                                 data_per_streamline=self.data_per_streamline)
        return sft

    def close(self):
        if self._uncompressed_folder_handle is not None:
            self._uncompressed_folder_handle.cleanup()
