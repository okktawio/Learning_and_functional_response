#include <math.h>
#include <float.h>
#include <string.h>

#include <python3.9/Python.h>
#include <python3.9/numpy/arrayobject.h>
#include <stdlib.h>

void rfl(double dx, double *sv, double *p, double *svo)
{
  /*
    Functional response with learning
    la = learning in a
    lH = learning in handling time
    Mnt = minimum handling time
    ha = host depletion
    No = N initial
    sv = state variables
  */
  double N = sv[0], a = sv[1], Ht = sv[2];
  double la = p[0], lH = p[1], Mnt = p[2], hd = p[3], No = p[4];
  double N2 = hd * N + (1 - hd) * No;
  double dN = (a * N2) / (1 + a * Ht * N2) * (hd * (N > 0) + (1 - hd));
  double da = la * (1 - a) * dN;
  double dH = lH * (Ht - Mnt) * dN;
  svo[0] = -dN * dx;
  svo[1] =  da * dx;
  svo[2] = -dH * dx;
}


void rk4_rfl(double dx, double p[], double N[])
{
  //dx /= 2.0;
  double N0[3], N1[3], N2[3], N3[3], N4[3];
  rfl(dx, N, p, N1);
  N0[0] = N[0] + N1[0] / 2.0;
  N0[1] = N[1] + N1[1] / 2.0;
  N0[2] = N[2] + N1[2] / 2.0; 
  rfl(dx, N0, p, N2);
  N0[0] = N[0] + N2[0] / 2.0;
  N0[1] = N[1] + N2[1] / 2.0;
  N0[2] = N[2] + N2[2] / 2.0; 
  rfl(dx,  N, p, N3);
  N0[0] = N[0] + N3[0];
  N0[1] = N[1] + N3[1];
  N0[2] = N[2] + N3[2]; 
  rfl(dx, N0, p, N4);
  //printf("No %6.4f N %6.4f dN %6.4f -> ", p[4], N[0], (N1[0] + 2 * N2[0] + 2 * N3[0] + N4[0]) / 6.);
  N[0] += (N1[0] + 2 * N2[0] + 2 * N3[0] + N4[0]) / 6.;
  N[1] += (N1[1] + 2 * N2[1] + 2 * N3[1] + N4[1]) / 6.;
  N[2] += (N1[2] + 2 * N2[2] + 2 * N3[2] + N4[2]) / 6.;
  //N[0] *= N[0] > 0;
}


static PyObject * rf_learning(PyObject * self, PyObject * args, PyObject * keywds)
{
  PyArrayObject *orf_params;
  npy_float64 tmax = 200;
  static char *kwlist[] = {"rf_params", "tmax", NULL};

  if (!PyArg_ParseTupleAndKeywords(args, keywds, "O|d", kwlist, &orf_params, &tmax))
    {
      printf ("ERROR de parametros\n");
      fflush (stdout);
      PyErr_SetString (PyExc_ValueError, "argument error");
    }

  npy_intp dims[2] = {(npy_intp) (tmax * 1024 + 1), 3};
  PyArrayObject * oNah = (PyArrayObject *) PyArray_SimpleNew (2, dims, PyArray_FLOAT64);
  //PyArrayObject * oa = (PyArrayObject *) PyArray_SimpleNew (1, dims, PyArray_FLOAT64);
  //PyArrayObject * oh = (PyArrayObject *) PyArray_SimpleNew (1, dims, PyArray_FLOAT64);
      
  npy_float64 * restrict Nah = ((npy_float64 *) (PyArray_DATA (oNah)));
  //npy_float64 * restrict a = ((npy_float64 *) (PyArray_DATA (oa)));
  //npy_float64 * restrict h = ((npy_float64 *) (PyArray_DATA (oh)));
  npy_float64 * restrict p = ((npy_float64 *) (PyArray_DATA (orf_params)));

  Nah[0] = 0;
  Nah[1] = p[5];
  Nah[2] = p[6];
  
  float delta = 1/1024.0;
  int j = 3;
  npy_float64 v_estado[3] = {p[4], p[5], p[6]};
  
  while(j < dims[0] * dims[1])
    {
      rk4_rfl(delta, p, v_estado);
      Nah[j] = p[4] - v_estado[0];
      Nah[j + 1] = v_estado[1];
      Nah[j + 2] = v_estado[2];
      //printf("%d %d %6.4f %6.4f %6.4f %6.4f %6.4f %6.4f %6.4f %6.4f\n", j, dims[0], p[3], p[4], p[5], p[6], Nah[j], Nah[j + 1], Nah[j + 2], v_estado[0]);
      j += 3;
    }
  fflush(stdout);
  return PyArray_Return(oNah);
}

static PyMethodDef crf_dinamica_methods[] =
  {
   {"rfl",   (PyCFunction) rf_learning, METH_VARARGS|METH_KEYWORDS, "respuesta funcional con aprendizaje"},
   {NULL, NULL, 0, NULL}
  };

static struct PyModuleDef Dinamicos =
  {
   PyModuleDef_HEAD_INIT,
   "crf_dinamica",               /* m_name */
   "Modelos Dinamicos de Respuesta Funcional",  /* m_doc */
   -1,                                 /* m_size */
   crf_dinamica_methods         /* m_methods */
  };


PyMODINIT_FUNC PyInit_crf_dinamica(void)
{
  import_array();
  return PyModule_Create(&Dinamicos);
}
