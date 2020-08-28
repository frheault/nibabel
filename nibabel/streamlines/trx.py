import json
import logging
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
# TODO fuse Move the save to class function
# TODO minimal docstring
# TODO rename data -> position
# TODO Eliminate the need for length as input (check time)
# TODO Catch the error if filename do not have a dtype extension (support bool?)


def _create_memmap(filename, mode='r', shape=(1,), dtype=np.float32, offset=0):
    if shape[0]:
        return np.memmap(filename, mode=mode, offset=offset,
                         shape=shape,  dtype=dtype)
    else:
        if not os.path.isfile(filename):
            f = open(filename, "wb")
            f.close
        return np.zeros(shape, dtype=dtype)


def load(input_obj, check_dpg=True):
    # Check if 0 streamlines, if yes then 0 points is expected (vice-versa)
    # 4x4 affine matrices should contains values (no all-zeros)
    # 3x1 dimensions array should contains values at each position (int)
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
                logging.info('File was compressed, call the close() '
                             'function before exiting.')
        else:
            trx = load_from_zip(input_obj)
    elif os.path.isdir(input_obj):
        trx = load_from_directory(input_obj)
    else:
        raise ValueError('File/Folder does not exist')

    # Example of robust check for metadata
    if check_dpg:
        for dpg in trx.data_per_group.keys():
            if dpg not in trx.groups.keys():
                raise ValueError('An undeclared group ({}) has '
                                 'data_per_group.'.format(dpg))

    return trx


def concatenate(trx_list, delete_dpp=False, delete_dps=False, delete_groups=False,
                check_space_attributes=True, preallocation=False):
    ref_trx = trx_list[0]

    if check_space_attributes:
        for curr_trx in trx_list[1:]:
            if not np.allclose(ref_trx.header['affine'], curr_trx.header['affine']) \
                    or not np.array_equal(ref_trx.header['dimensions'], curr_trx.header['dimensions']):
                raise ValueError('Wrong space attributes.')

    if preallocation and not delete_groups:
        raise ValueError('Groups are variables, cannot be handled with '
                         'preallocation')

    # Verifying the validity of fixed-size arrays, coherence between inputs
    for curr_trx in trx_list[1:]:
        for key in curr_trx.data_per_point.keys():
            if key not in ref_trx.data_per_point.keys():
                if not delete_dpp:
                    logging.debug('{} dpp key does not exist in all TrxFile.'.format(
                        key))
                    raise ValueError('TrxFile must be sharing identical dpp '
                                     'keys.')
            elif ref_trx.data_per_point[key]._data.dtype \
                    != curr_trx.data_per_point[key]._data.dtype:
                logging.debug('{} dpp key is not declared with the same dtype '
                              'in all TrxFile.'.format(key))
                raise ValueError('Shared dpp key, has different dtype.')

    for curr_trx in trx_list[1:]:
        for key in curr_trx.data_per_streamline.keys():
            if key not in ref_trx.data_per_streamline.keys():
                if not delete_dps:
                    logging.debug('{} dps key does not exist in all TrxFile.'.format(
                        key))
                    raise ValueError('TrxFile must be sharing identical dps '
                                     'keys.')
            elif ref_trx.data_per_streamline[key].dtype \
                    != curr_trx.data_per_streamline[key].dtype:
                logging.debug('{} dps key is not declared with the same dtype '
                              'in all TrxFile.'.format(key))
                raise ValueError('Shared dps key, has different dtype.')

    all_groups_len = {}
    all_groups_dtype = {}
    # Variable-size arrays do not have to exist in all TrxFile
    if not delete_groups:
        for trx_1 in trx_list:
            for group_key in trx_1.groups.keys():
                # Concatenating groups together
                if group_key in all_groups_len:
                    all_groups_len[group_key] += len(trx_1.groups[group_key])
                else:
                    all_groups_len[group_key] = len(trx_1.groups[group_key])
                    all_groups_dtype[group_key] = trx_1.groups[group_key].dtype

                if len(trx_1.data_per_group.keys()) > 0:
                    logging.warning('TrxFile contains data_per_group, this '
                                    'information cannot be easily concatenated, it '
                                    'will be deleted.')

                # data_per_group cannot be easily 'fused', so they are discarded
                for trx_2 in trx_list:
                    if trx_1.groups[group_key].dtype \
                                != trx_2.groups[group_key].dtype:
                        logging.debug('{} group key is not declared with '
                                      'the same dtype in all TrxFile.'.format(
                                                key))
                        raise ValueError('Shared group key, has different dtype.')

    # Once the checks are done, actually concatenate
    nbr_points = 0
    nbr_streamlines = 0
    to_concat_list = trx_list[1:] if preallocation else trx_list
    for curr_trx in to_concat_list:
        curr_strs_len, curr_pts_len = curr_trx._get_real_size()
        nbr_streamlines += curr_strs_len
        nbr_points += curr_pts_len

    if not preallocation:
        new_trx = TrxFile(nbr_points=nbr_points, nbr_streamlines=nbr_streamlines,
                          init_as=ref_trx)
        tmp_dir = new_trx._uncompressed_folder_handle.name

        # When memory is allocated on the spot, groups and data_per_group can
        # be concatenated together
        for group_key in all_groups_len.keys():
            if not os.path.isdir(os.path.join(tmp_dir, 'groups/')):
                os.mkdir(os.path.join(tmp_dir, 'groups/'))
            dtype = all_groups_dtype[group_key]
            group_filename = os.path.join(tmp_dir, 'groups/'
                                          '{}.{}'.format(group_key,
                                                         dtype.name))
            new_trx.groups[group_key] = _create_memmap(group_filename, mode='w+',
                                                       shape=(all_groups_len[group_key],),
                                                       dtype=dtype)
            if delete_groups:
                continue
            pos = 0
            count = 0
            for curr_trx in trx_list:
                curr_len = len(curr_trx.groups[group_key])
                new_trx.groups[group_key][pos:pos+curr_len] = \
                    curr_trx.groups[group_key] + count
                pos += curr_len
                count += curr_trx.header['nbr_streamlines']

        strs_end, pts_end = 0, 0
    else:
        new_trx = ref_trx
        strs_end, pts_end = new_trx._get_real_size()

    for curr_trx in to_concat_list:
        # Copy the TrxFile fixed-size info (the right chunk)
        strs_end, pts_end = new_trx._copy_fixed_arrays_from(curr_trx,
                                                            strs_start=strs_end,
                                                            pts_start=pts_end)

    return new_trx


def load_from_zip(filename):
    with zipfile.ZipFile(filename, mode='r') as zf:
        with zf.open('header.json') as zf_header:
            data = zf_header.read()
            header = json.loads(data)

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
    return TrxFile._create_trx_from_pointer(header, files_pointer_size,
                                            root_zip=filename)


def load_from_directory(directory):
    directory = os.path.abspath(directory)
    with open(os.path.join(directory, 'header.json')) as header:
        header = json.load(header)

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

    return TrxFile._create_trx_from_pointer(header, files_pointer_size, root=directory)


def save(trx, filename):
    trx.resize()
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
            to_dump = trx.data_per_point[dpp_key]._data
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

        if len(trx.groups.keys()) > 0:
            os.mkdir(os.path.join(tmp_dir, 'groups/'))
        for group_key in trx.groups.keys():
            to_dump = trx.groups[group_key]
            to_dump.tofile(os.path.join(tmp_dir,
                                        'groups/{}.{}'.format(group_key,
                                                              to_dump.dtype.name)))
            if group_key not in trx.data_per_group:
                continue
            for dpg_key in trx.data_per_group[group_key].keys():
                # Creates 'dpg/' only if required
                if not os.path.isdir(os.path.join(tmp_dir, 'dpg/')):
                    os.mkdir(os.path.join(tmp_dir, 'dpg/'))
                os.mkdir(os.path.join(tmp_dir, 'dpg/', group_key))
                to_dump = trx.data_per_group[group_key][dpg_key]
                to_dump.tofile(os.path.join(tmp_dir,
                                            'dpg/{}/{}.{}'.format(group_key,
                                                                  dpg_key,
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
        if init_as is not None:
            affine = init_as.header['affine']
            dimensions = init_as.header['dimensions']
        elif reference is not None:
            affine, dimensions, _, _ = get_reference_info(reference)
        else:
            logging.debug('No reference provided, using blank space '
                          'attributes, please update them later.')
            affine = np.eye(4)
            dimensions = np.array([1, 1, 1], dtype=np.uint16)

        if nbr_points is None and nbr_streamlines is None:
            if init_as is not None:
                raise ValueError('Cant use init_as without declaring '
                                 'nbr_points AND nbr_streamlines')
            logging.debug('Intializing empty TrxFile.')
            self.header = {}
            self.streamlines = ArraySequence()
            self.groups = {}
            self.data_per_streamline = {}
            self.data_per_point = {}
            self.data_per_group = {}
            self._uncompressed_folder_handle = None

            nbr_points = 0
            nbr_streamlines = 0

        elif nbr_points is not None and nbr_streamlines is not None:
            logging.debug('Preallocating TrxFile with size {} streamlines'
                          'and {} points.'.format(nbr_streamlines, nbr_points))
            trx = self._initialize_empty_trx(nbr_streamlines, nbr_points,
                                             init_as=init_as)
            self.__dict__ = trx.__dict__
        else:
            raise ValueError('You must declare both nbr_points AND '
                             'nbr_streamlines')

        self.header['affine'] = affine
        self.header['dimensions'] = dimensions
        self.header['nbr_points'] = nbr_points
        self.header['nbr_streamlines'] = nbr_streamlines

    @staticmethod
    def _initialize_empty_trx(nbr_streamlines, nbr_points, init_as=None):
        trx = TrxFile()
        tmp_dir = tempfile.TemporaryDirectory()
        logging.info('Temporary folder for memmap: {}'.format(tmp_dir.name))

        if init_as is not None:
            data_dtype = init_as.streamlines._data.dtype
            offsets_dtype = init_as.streamlines._offsets.dtype
            lengths_dtype = init_as.streamlines._lengths.dtype
        else:
            data_dtype = np.dtype(np.float16)
            offsets_dtype = np.dtype(np.uint64)
            lengths_dtype = np.dtype(np.uint32)

        logging.debug('Initializing data with dtype:    {}'.format(
            data_dtype.name))
        logging.debug('Initializing offsets with dtype: {}'.format(
            offsets_dtype.name))
        logging.debug('Initializing lengths with dtype: {}'.format(
            lengths_dtype.name))

        # A TrxFile without init_as only contain the essential arrays
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

        # Only the structure of fixed-size arrays is copied
        if init_as is not None:
            if len(init_as.data_per_point.keys()) > 0:
                os.mkdir(os.path.join(tmp_dir.name, 'dpp/'))
            if len(init_as.data_per_streamline.keys()) > 0:
                os.mkdir(os.path.join(tmp_dir.name, 'dps/'))

            for dpp_key in init_as.data_per_point.keys():
                dtype = init_as.data_per_point[dpp_key]._data.dtype
                dpp_filename = os.path.join(tmp_dir.name, 'dpp/'
                                            '{}.{}'.format(dpp_key, dtype.name))
                logging.debug('Initializing {} (dpp) with dtype: {}'.format(
                    dpp_key, dtype.name))
                trx.data_per_point[dpp_key] = ArraySequence()
                trx.data_per_point[dpp_key]._data = _create_memmap(dpp_filename,
                                                                   mode='w+',
                                                                   shape=(nbr_points, 1),
                                                                   dtype=dtype)
                trx.data_per_point[dpp_key]._offsets = trx.streamlines._offsets
                trx.data_per_point[dpp_key]._lengths = trx.streamlines._lengths

            for dps_key in init_as.data_per_streamline.keys():
                dtype = init_as.data_per_streamline[dps_key].dtype
                logging.debug('Initializing {} (dps) with dtype: {}'.format(
                    dps_key, dtype.name))

                dps_filename = os.path.join(tmp_dir.name, 'dps/'
                                            '{}.{}'.format(dps_key, dtype.name))
                trx.data_per_streamline[dps_key] = _create_memmap(dps_filename,
                                                                  mode='w+',
                                                                  shape=(nbr_streamlines,),
                                                                  dtype=dtype)

        trx._uncompressed_folder_handle = tmp_dir

        return trx

    @staticmethod
    def _create_trx_from_pointer(header, dict_pointer_size,
                                 root_zip=None, root=None):
        trx = TrxFile()
        trx.header = header

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

            if base == 'data' and folder == '':
                if size != trx.header['nbr_points']*3:
                    raise ValueError('Wrong data size.')
                data = _create_memmap(filename, mode='r+',
                                      offset=mem_adress,
                                      shape=(trx.header['nbr_points'], 3),
                                      dtype=ext[1:])
            elif base == 'offsets' and folder == '':
                if size != trx.header['nbr_streamlines']:
                    raise ValueError('Wrong offsets size.')
                offsets = _create_memmap(filename, mode='r+',
                                         offset=mem_adress,
                                         shape=(trx.header['nbr_streamlines'],),
                                         dtype=ext[1:])
            elif base == 'lengths' and folder == '':
                if size != trx.header['nbr_streamlines']:
                    raise ValueError('Wrong lengths size.')
                lengths = _create_memmap(filename, mode='r+',
                                         offset=mem_adress,
                                         shape=(trx.header['nbr_streamlines'],),
                                         dtype=ext[1:])
            elif folder == 'dps':
                if size != trx.header['nbr_streamlines']:
                    raise ValueError('Wrong dps size.')
                trx.data_per_streamline[base] = _create_memmap(
                    filename, mode='r+', offset=mem_adress,
                    shape=(trx.header['nbr_streamlines'],), dtype=ext[1:])
            elif folder == 'dpp':
                if size != trx.header['nbr_points']:
                    raise ValueError('Wrong dpp size.')
                trx.data_per_point[base] = _create_memmap(
                    filename, mode='r+', offset=mem_adress,
                    shape=(trx.header['nbr_points'], 1), dtype=ext[1:])
            elif folder.startswith('dpg'):
                if size != 1:
                    raise ValueError('Wrong dpg size.')
                data_name = os.path.basename(base)
                sub_folder = os.path.basename(folder)
                if sub_folder not in trx.data_per_group:
                    trx.data_per_group[sub_folder] = {}
                trx.data_per_group[sub_folder][data_name] = _create_memmap(
                    filename, mode='r+', offset=mem_adress,
                    shape=(1,), dtype=ext[1:])
            elif folder == 'groups':
                trx.groups[base] = _create_memmap(filename, mode='r+',
                                                  offset=mem_adress,
                                                  shape=(size,),
                                                  dtype=ext[1:])
            else:
                raise ValueError('Incorrect structure, cannot parse.')

        # All essential array must be declared
        if data is not None \
            and offsets is not None \
                and lengths is not None:
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

    def _get_real_size(self):
        if np.any(self.streamlines._offsets):
            strs_end = int(np.nonzero(self.streamlines._offsets)[0][-1]+1)
            pts_end = int(np.sum(self.streamlines._lengths[0:strs_end]))
        else:
            strs_end = 0
            pts_end = 0

        return strs_end, pts_end

    def resize(self, nbr_streamlines=None, nbr_points=None, delete_dpg=False):
        strs_end, pts_end = self._get_real_size()
        if nbr_streamlines is not None and nbr_streamlines < strs_end:
            strs_end = nbr_streamlines
            logging.info('Resizing (down) memmap, less streamlines than it '
                         'actually contains.')
            if nbr_points is not None:
                pts_end = int(np.sum(self.streamlines._lengths[0:nbr_streamlines]))
                logging.warning('Keeping the appropriate points count for '
                                'consistency, overwritting provided parameters.')

        if nbr_streamlines is None:
            nbr_streamlines = strs_end
            if nbr_streamlines == self.header['nbr_streamlines']:
                logging.debug('TrxFile of the right size, no resizing.')
                return

        if nbr_points is None:
            nbr_points = pts_end
        elif nbr_points < pts_end:
            logging.warning('Cannot resize (down) points for consistency.')

        trx = self._initialize_empty_trx(nbr_streamlines, nbr_points, init_as=self)
        trx.header['affine'] = self.header['affine']
        trx.header['dimensions'] = self.header['dimensions']
        trx.header['nbr_points'] = nbr_points
        trx.header['nbr_streamlines'] = nbr_streamlines

        logging.info('Resizing streamlines from size {} to {}'.format(
            len(self.streamlines), nbr_streamlines))
        logging.info('Resizing points from size {} to {}'.format(
            len(self.streamlines._data), nbr_points))

        # Copy the fixed-sized info from the original TrxFile to the new
        # (resized) one.
        trx._copy_fixed_arrays_from(self)

        tmp_dir = trx._uncompressed_folder_handle.name
        if len(self.groups.keys()) > 0:
            os.mkdir(os.path.join(tmp_dir, 'groups/'))

        for group_key in self.groups.keys():
            group_dtype = self.groups[group_key].dtype
            group_name = os.path.join(tmp_dir, 'groups/',
                                      '{}.{}'.format(group_key,
                                                     group_dtype.name))
            ori_len = self.groups[group_key]
            tmp = self.groups[group_key][self.groups[group_key] < strs_start]
            trx.groups[group_key] = _create_memmap(group_name, mode='w+',
                                                   shape=(len(tmp),),
                                                   dtype=group_dtype)
            logging.debug('{} group went from {} items to {}'.format(group_key,
                                                                     ori_len,
                                                                     len(tmp)))
            trx.groups[group_key][:] = tmp

            if not delete_dpg:
                if len(trx.data_per_group.keys()) > 0:
                    os.mkdir(os.path.join(tmp_dir, 'dpg/'))

                if group_key not in trx.data_per_group:
                    self.data_per_group[group_key] = {}
                    for dpg_key in self.data_per_group[group_key].keys():
                        dpg_dtype = self.data_per_group[group_key][dpg_key].dtype
                        dpg_name = os.path.join(tmp_dir, 'dpg/', group_key,
                                                '{}.{}'.format(dpg_key,
                                                               dpg_dtype.name))
                        if dpg_key not in trx.data_per_group[group_key]:
                            trx.data_per_group[group_key] = {}
                        trx.data_per_group[group_key][dpg_key] = \
                            _create_memmap(dpg_name, mode='w+', shape=(1,),
                                           dtype=dpg_dtype)

                        trx.data_per_group[group_key][dpg_key][:] \
                            = self.self.data_per_group[group_key][dpg_key]

        self.close()
        self.__dict__ = trx.__dict__

    def _copy_fixed_arrays_from(self, trx, strs_start=0, pts_start=0):
        curr_strs_len, curr_pts_len = trx._get_real_size()
        strs_end = strs_start + curr_strs_len
        pts_end = pts_start + curr_pts_len

        # Mandatory arrays
        self.streamlines._data[pts_start:pts_end] \
            = trx.streamlines._data[0:curr_pts_len]
        self.streamlines._offsets[strs_start:strs_end] = \
            trx.streamlines._offsets[0:curr_strs_len] + pts_start
        self.streamlines._lengths[strs_start:strs_end] = \
            trx.streamlines._lengths[0:curr_strs_len]

        # Optional fixed-sized arrays
        for dpp_key in self.data_per_point.keys():
            self.data_per_point[dpp_key]._data[pts_start:pts_end] = \
                trx.data_per_point[dpp_key]._data[0:curr_pts_len]
            self.data_per_point[dpp_key]._offsets[strs_start:strs_end] = \
                trx.streamlines._offsets[0:curr_strs_len] + pts_start
            self.data_per_point[dpp_key]._lengths[strs_start:strs_end] = \
                trx.streamlines._lengths[0:curr_strs_len]

        for dps_key in self.data_per_streamline.keys():
            self.data_per_streamline[dps_key][strs_start:strs_end] \
                = trx.data_per_streamline[dps_key][0:curr_strs_len]

        return strs_end, pts_end

    def append(self, trx, buffer_size=0):
        strs_end, pts_end = self._get_real_size()

        nbr_streamlines = strs_end + trx.header['nbr_streamlines']
        nbr_points = pts_end + trx.header['nbr_points']
        if self.header['nbr_streamlines'] < nbr_streamlines \
                or self.header['nbr_points'] < nbr_points:
            self.resize(nbr_streamlines=nbr_streamlines+buffer_size,
                        nbr_points=nbr_points+buffer_size*100)
            _ = concatenate([self, trx], preallocation=True,
                            delete_groups=True)
        else:
            trx = concatenate([self, trx], preallocation=False,
                              delete_groups=False)
            self.close()
            self.__dict__ = trx.__dict__

    @ staticmethod
    def from_sft(sft):
        trx = TrxFile()
        trx.header = {'dimensions': sft.dimensions.tolist(),
                      'affine': sft.affine.tolist(),
                      'nbr_points': len(sft.streamlines._data),
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
