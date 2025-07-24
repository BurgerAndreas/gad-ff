import numpy as np
from scipy.linalg import block_diag
from .math_utils import orthogonalize, conjugate_orthogonalize


def isblock(obj):
    return hasattr(obj, "matlist")


class block_matrix(object):

    def __init__(self, matlist, cnorms=None):
        self.matlist = matlist
        if cnorms is None:
            cnorms = np.zeros((self.shape[1], 1))
        self.cnorms = cnorms

    def __repr__(self):
        lines = [" block matrix: # blocks = {}".format(self.num_blocks)]
        count = 0
        for m in self.matlist:
            lines.append(str(m))
            count += 1
            if count > 10:
                nifty.logger.debug("truncating printout")
                break
        return "\n".join(lines)

    @staticmethod
    def full_matrix(A):
        return block_diag(*A.matlist)

    @property
    def num_blocks(self):
        return len(self.matlist)

    # IDEA: everywhere a dot product of DLC is done, use the conjugate
    # dot product, also use the conjugate_orthogonalize to orthogonalize
    @staticmethod
    def project_conjugate_constraint(BM, constraints, G):
        def ov(vi, vj):
            return np.linalg.multi_dot([vi, G, vj])

        # the constraints need to be orthonormalized on G
        constraints = conjugate_orthogonalize(constraints, G)

        # (a) need to zero some segments (corresponding to the blocks of
        #  Vecs) of the constraints if their magnitude is small
        s = 0
        for block in BM.matlist:
            size = len(block)
            e = s + size
            for constraint in constraints.T:
                if (constraint[s:e] == 0.0).all():
                    pass
                elif np.linalg.norm(constraint[s:e]) < 1.0e-3:
                    constraint[s:e] = np.zeros(size)
            s = e

        # (b) renormalizing the constraints on the surface G
        norms = np.sqrt((ov(constraints.T, constraints).sum(axis=0, keepdims=True)))
        constraints = constraints / norms
        # nifty.logger.debug('constraints after renormalizing')
        # nifty.logger.debug(constraints.T)

        # (c) need to save the magnitude of the constraints in each segment since they
        # will be renormalized for each block
        cnorms = np.zeros((BM.shape[1], constraints.shape[1]))
        sr = 0
        sc = 0
        newblocks = []
        for block in BM.matlist:
            size_r = block.shape[0]
            size_c = block.shape[1]
            er = sr + size_r
            ec = sc + size_c
            flag = False
            tmpc = []
            for count, constraint in enumerate(constraints.T):

                # CRA 81219 what to do here? mag of real or g-space?
                # mag = np.linalg.norm(constraint[sr:er])
                mag = np.sqrt(
                    np.linalg.multi_dot(
                        [constraint[sr:er], G[sr:er, sr:er], constraint[sr:er]]
                    )
                )

                # concatenating the block to each constraint if the constraint is greater than parameter
                if mag > 1.0e-3:
                    cnorms[sc + count, count] = mag
                    tmpc.append(constraint[sr:er] / mag)
                    flag = True
            if flag:
                tmpc = np.asarray(tmpc).T
                if len(tmpc) != len(block):
                    raise RuntimeError
                newblocks.append(np.hstack((tmpc, block)))
            else:
                newblocks.append(block)
            sr = er
            sc = ec

        # (d) orthogonalize each block
        ans = []
        sr = 0
        sc = 0
        count = 0
        for nb, ob in zip(newblocks, BM.matlist):
            size_c = ob.shape[1]
            size_r = block.shape[0]
            er = sr + size_r
            ec = sc + size_c
            num_c = 0
            flag = False
            for c in cnorms.T:
                if any(c[sc:ec] != 0.0):
                    num_c += 1
                    flag = True
            if flag:
                ans.append(conjugate_orthogonalize(nb, G[sr:er, sr:er], num_c))
            else:
                ans.append(conjugate_orthogonalize(nb, G[sr:er, sr:er]))
                # ans.append(ob)
            sc = ec
            sr = er
            count += 1
        return block_matrix(ans, cnorms)

    # TODO 8/10/2019 write a detailed explanation for this method
    @staticmethod
    def project_constraint(BM, constraints):
        assert len(constraints) == len(BM)

        # (a) need to zero some segments (corresponding to the blocks of Vecs) of the constraints if their magnitude is small
        s = 0
        for block in BM.matlist:
            size = len(block)
            e = s + size
            for constraint in constraints.T:
                if (constraint[s:e] == 0.0).all():
                    pass
                elif np.linalg.norm(constraint[s:e]) < 1.0e-2:
                    constraint[s:e] = np.zeros(size)
            s = e

        # (b) renormalizing the constraints
        norms = np.sqrt((constraints * constraints).sum(axis=0, keepdims=True))
        # nifty.logger.debug('norms')
        # nifty.logger.debug(norms)
        constraints = constraints / norms

        # (c) need to save the magnitude of the constraints in each segment since they
        # will be renormalized for each block
        cnorms = np.zeros((BM.shape[1], constraints.shape[1]))
        sr = 0
        sc = 0
        newblocks = []
        for block in BM.matlist:
            size_r = block.shape[0]
            size_c = block.shape[1]
            er = sr + size_r
            ec = sc + size_c
            flag = False
            tmpc = []
            for count, constraint in enumerate(constraints.T):
                mag = np.linalg.norm(constraint[sr:er])
                # (d) concatenating the block to each constraint if the constraint is greater than parameter
                if mag > 1.0e-2:
                    cnorms[sc + count, count] = mag
                    tmpc.append(constraint[sr:er] / mag)
                    flag = True
            if flag:
                tmpc = np.asarray(tmpc).T
                if len(tmpc) != len(block):
                    raise RuntimeError
                newblocks.append(np.hstack((tmpc, block)))
            else:
                newblocks.append(block)
            sr = er
            sc = ec

        assert len(newblocks) == len(BM.matlist), "not proper lengths for zipping"

        # NEW
        # orthogonalize each sub block
        # nifty.logger.debug(" Beginning to orthogonalize each sub block")
        ans = []
        sc = 0
        count = 0
        for nb, ob in zip(newblocks, BM.matlist):
            # nifty.logger.debug("On block %d" % count)
            size_c = ob.shape[1]
            ec = sc + size_c
            num_c = 0
            flag = False
            for c in cnorms.T:
                # if (c[sc:ec]!=0.).any():
                if any(c[sc:ec] != 0.0):
                    num_c += 1
                    # nifty.logger.debug('block %d mag %.4f' %(count,np.linalg.norm(c[sc:ec])))
                    # nifty.logger.debug(c[sc:ec])
                    # nifty.logger.debug('num_c=%d' %num_c)
                    flag = True
            # nifty.logger.debug(flag)
            if flag:
                # nifty.logger.debug(" orthogonalizing sublock {} with {} constraints".format(count,num_c))
                # nifty.logger.debug(ob.shape)
                # nifty.logger.debug(nb.shape)
                try:
                    a = orthogonalize(nb, num_c)
                    # nifty.logger.debug("result {}".format(a.shape))
                except:
                    nifty.logger.debug(" what is happening")
                    nifty.logger.debug("nb")
                    nifty.logger.debug(nb)
                    nifty.logger.debug(nb.shape)
                    nifty.logger.debug(num_c)
                    nifty.logger.debug("ob")
                    nifty.logger.debug(ob)
                ans.append(a)
                # ans.append(orthogonalize(nb,num_c))
            else:
                # nifty.logger.debug(" appending old block without constraints")
                ans.append(ob)
            sc = ec
            count += 1

        return block_matrix(ans, cnorms)

        # (d) concatenating the block to each constraint if the constraint is non-zero
        # sr=0
        # newblocks=[]
        # for block in BM.matlist:
        #    size_r=block.shape[0]
        #    er=sr+size_r
        #    flag=False
        #    tmpc = []
        #    for constraint in constraints.T:
        #        #if (constraint[s:e]!=0.).all():
        #        mag = np.linalg.norm(constraint[sr:er])
        #        if mag>0.:
        #            tmpc.append(constraint[sr:er]/mag)
        #            flag=True
        #    if flag==True:
        #        tmpc = np.asarray(tmpc).T
        #        if len(tmpc)!=len(block):
        #            #nifty.logger.debug(tmpc.shape)
        #            #nifty.logger.debug(block.shape)
        #            #nifty.logger.debug('start %i end %i' %(s,e))
        #            raise RuntimeError
        #        newblocks.append(np.hstack((tmpc,block)))
        #    else:
        #        newblocks.append(block)
        #    sr=er

        # nifty.logger.debug('cnorms')
        # nifty.logger.debug(cnorms[np.nonzero(cnorms)[0]])
        # return block_matrix(newblocks,cnorms)

    @staticmethod
    def qr(BM):  # only return the Q part
        # nifty.logger.debug("before qr")
        # nifty.logger.debug(BM)
        ans = []
        for A in BM.matlist:
            Q, R = np.linalg.qr(A)
            indep = np.where(np.abs(R.diagonal()) > min_tol)[0]
            ans.append(Q[:, indep])
            if len(indep) > A.shape[1]:
                nifty.logger.debug(" the basis dimensions are too large.")
                raise RuntimeError
            # tmp = np.dot(Q,R)
            # nifty.logger.debug(tmp.shape)
            # nifty.logger.debug("r,q shape")
            # nifty.logger.debug(R.shape)
            # pvec1d(R[-1,:])
            # ans.append(Q[:,:BM.shape[1]-BM.cnorms.shape[1]])
            # m=A.shape[1]
            # nifty.logger.debug(R)
            # for i in range(BM.cnorms.shape[1]):
            #    if np.linalg.norm(R[-1,:])<1.e-3:
            #        m-=1
            # ans.append(Q[:,:m])
        return block_matrix(ans, BM.cnorms)
        # return block_matrix( [ np.linalg.qr(A)[0] for A in BM.matlist ], BM.cnorms)

    @staticmethod
    def diagonal(BM):
        la = [np.diagonal(A) for A in BM.matlist]
        return np.concatenate(la)

    @staticmethod
    def gram_schmidt(BM):
        ans = []
        sc = 0
        for i, block in enumerate(BM.matlist):
            size_c = block.shape[1]
            ec = sc + size_c
            num_c = 0
            for c in BM.cnorms.T:
                if any(c[sc:ec] != 0.0):
                    num_c += 1
                    nifty.logger.debug(
                        "block %d mag %.4f" % (i, np.linalg.norm(c[sc:ec]))
                    )
                    nifty.logger.debug(c[sc:ec])
                    nifty.logger.debug("num_c=%d" % num_c)
            ans.append(orthogonalize(block, num_c))
            sc = ec
        return block_matrix(ans, BM.cnorms)

    @staticmethod
    def eigh(BM):
        eigenvalues = []
        eigenvectors = []
        for block in BM.matlist:
            e, v = np.linalg.eigh(block)
            eigenvalues.append(e)
            eigenvectors.append(v)
        return np.concatenate(eigenvalues), block_matrix(eigenvectors)

    @staticmethod
    def zeros_like(BM):
        return block_matrix([np.zeros_like(A) for A in BM.matlist])

    def __add__(self, rhs):
        nifty.logger.debug("adding")
        if isinstance(rhs, self.__class__):
            nifty.logger.debug("adding block matrices!")
            assert self.shape == rhs.shape
            return block_matrix([A + B for A, B in zip(self.matlist, rhs.matlist)])
        elif isinstance(rhs, float) or isinstance(rhs, int):
            return block_matrix([A + rhs for A in self.matlist])
        else:
            raise NotImplementedError

    def __radd__(self, lhs):
        return self.__add__(lhs)

    def __mul__(self, rhs):
        if isinstance(rhs, self.__class__):
            assert self.shape == rhs.shape
            return block_matrix([A * B for A, B in zip(self.matlist, rhs.matlist)])
        elif isinstance(rhs, float) or isinstance(rhs, int):
            return block_matrix([A * rhs for A in self.matlist])
        else:
            raise NotImplementedError

    def __rmul__(self, lhs):
        return self.__mul__(lhs)

    def __len__(self):  # size along first axis
        return np.sum([len(A) for A in self.matlist])

    def __truediv__(self, rhs):
        if isinstance(rhs, self.__class__):
            assert self.shape == rhs.shape
            return block_matrix([A / B for A, B in zip(self.matlist, rhs.matlist)])
        elif isinstance(rhs, float) or isinstance(rhs, int):
            return block_matrix([A / rhs for A in self.matlist])
        elif isinstance(rhs, np.ndarray):
            answer = []
            s = 0
            for block in self.matlist:
                e = block.shape[1] + s
                answer.append(block / rhs[s:e])
                s = e
            return block_matrix(answer)
        else:
            raise NotImplementedError

    @property
    def shape(self):
        tot = (0, 0)
        for a in self.matlist:
            tot = tuple(map(sum, zip(a.shape, tot)))
        return tot

    @staticmethod
    def transpose(A):
        return block_matrix([A.T for A in A.matlist])

    @staticmethod
    def dot(left, right):
        def block_vec_dot(block, vec):
            if vec.ndim == 2 and vec.shape[1] == 1:
                vec = vec.flatten()
            # if block.cnorms is None:
            s = 0
            result = []
            for A in block.matlist:
                e = s + np.shape(A)[1]
                result.append(np.dot(A, vec[s:e]))
                s = e
            return np.reshape(np.concatenate(result), (-1, 1))

        def vec_block_dot(vec, block, **kwargs):
            if vec.ndim == 2 and vec.shape[1] == 1:
                vec = vec.flatten()
            # if block.cnorms is None:
            s = 0
            result = []
            for A in block.matlist:
                e = s + np.shape(A)[1]
                result.append(np.dot(vec[s:e], A))
                s = e
            return np.reshape(np.concatenate(result), (-1, 1))

        if isinstance(left, np.ndarray) and left.ndim == 1:
            left = np.reshape(left, (-1, 1))
        if isinstance(right, np.ndarray) and right.ndim == 1:
            right = np.reshape(right, (-1, 1))

        # (1) both are block matrices
        if isblock(left) and isblock(right):
            return block_matrix(
                [np.dot(A, B) for A, B in zip(left.matlist, right.matlist)]
            )
        # (2) left is np.ndarray with a vector shape
        elif isinstance(left, np.ndarray) and left.shape[1] == 1 and isblock(right):
            return vec_block_dot(left, right)
        # (3) right is np.ndarray with a vector shape
        elif isinstance(right, np.ndarray) and right.shape[1] == 1 and isblock(left):
            return block_vec_dot(left, right)
        # (4) l/r is a matrix
        elif isinstance(left, np.ndarray) and left.shape[1] > 1:
            #
            # [ A | B ] [ C 0 ] = [ AC BD ]
            #           [ 0 D ]
            sc = 0
            tmp_ans = []
            for A in right.matlist:
                ec = sc + A.shape[0]
                tmp_ans.append(np.dot(left[:, sc:ec], A))
                sc = ec
            dot_product = np.hstack(tmp_ans)
            return dot_product

        elif isinstance(right, np.ndarray) and right.shape[1] > 1:
            #
            # [ A | 0 ] [ C ] = [ AC ]
            # [ 0 | B ] [ D ]   [ BD ]
            sc = 0
            tmp_ans = []
            for A in left.matlist:
                ec = sc + A.shape[1]
                tmp_ans.append(np.dot(A, right[sc:ec, :]))
                sc = ec
            dot_product = np.vstack(tmp_ans)
            return dot_product
        else:
            raise NotImplementedError


# if __name__=="__main__":

# A = [np.array([[1,2],[3,4]]), np.array([[5,6],[7,8]])]
# B = [np.array([[1,2],[3,4]]), np.array([[5,6],[7,8]])]
# Ab = bm(A)
# Bb = bm(B)
#
# nifty.logger.debug("A")
# nifty.logger.debug(Ab)
#
# nifty.logger.debug("B")
# nifty.logger.debug(Bb)
#
# test 1
# nifty.logger.debug("test 1 adding block matrices")
# Cb = Ab+Bb
# nifty.logger.debug(Cb)
#
# nifty.logger.debug("test 2 adding block matrix and float")
# Db = Ab+2
# nifty.logger.debug(Db)
#
# nifty.logger.debug("test 3 reversing order of addition")
# Eb = 2+Ab
# nifty.logger.debug(Eb)
#
# nifty.logger.debug("test 4 block multiplication")
# Fb = Ab*Bb
# nifty.logger.debug(Fb)
#
# nifty.logger.debug("test 5 block multiplication by scalar")
# Gb = Ab*2
# nifty.logger.debug(Gb)
#
# nifty.logger.debug("test 6 reverse block mult by scalar")
# Hb = 2*Ab
# nifty.logger.debug(Hb)
#
# nifty.logger.debug("test 7 total len")
# nifty.logger.debug(len(Hb))
#
# nifty.logger.debug("test 8 shape")
# nifty.logger.debug(Hb.shape)
#
# nifty.logger.debug("test dot product with block matrix")
# Ib = bm.dot(Ab,Bb)
# nifty.logger.debug(Ib)
#
# nifty.logger.debug("test dot product with np vector")
# Jb = bm.dot(Ab,np.array([1,2,3,4]))
# nifty.logger.debug(Jb)
#
# nifty.logger.debug("Test dot product with np 2d vector shape= (x,1)")
# a = np.array([[1,2,3,4]]).T
# Kb = bm.dot(Ab,a)
# nifty.logger.debug(Kb)
#
# nifty.logger.debug("test dot product with non-block array")
# fullmat = np.random.randint(5,size=(4,4))
# nifty.logger.debug(" full mat to mult")
# nifty.logger.debug(fullmat)
# A = [np.array([[1,2,3],[4,5,6]]), np.array([[7,8,9],[10,11,12]])]
# Ab = bm(A)
# nifty.logger.debug(" Ab")
# nifty.logger.debug(bm.full_matrix(Ab))
# nifty.logger.debug('result')
# Mb = np.dot(fullmat,bm.full_matrix(Ab))
# nifty.logger.debug(Mb)
# Lb = bm.dot(fullmat,Ab)
# nifty.logger.debug('result of dot product with full mat')
# nifty.logger.debug(Lb)
# nifty.logger.debug(Lb == Mb)
#
# nifty.logger.debug("test dot product with non-block array")
# nifty.logger.debug(" full mat to mult")
# nifty.logger.debug(fullmat)
# nifty.logger.debug(" Ab")
# nifty.logger.debug(bm.full_matrix(Ab))
# nifty.logger.debug('result')
# A = [ np.array([[1,2],[3,4],[5,6]]),np.array([[7,8],[9,10],[11,12]])]
# Ab = bm(A)
# nifty.logger.debug(Ab.shape)
# nifty.logger.debug(fullmat.shape)
# Mb = np.dot(bm.full_matrix(Ab),fullmat)
# nifty.logger.debug(Mb)
# Lb = bm.dot(Ab,fullmat)
# nifty.logger.debug('result of dot product with full mat')
# nifty.logger.debug(Lb)
# nifty.logger.debug(Lb == Mb)
#
