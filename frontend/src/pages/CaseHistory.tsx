import React, { useEffect, useState } from 'react';
import { NavLink, useNavigate } from 'react-router-dom';
import { api } from '../api';
import { CaseSummary, CaseDetailType, AuditEvent } from '../types';
import {
  FolderKanban,
  FileSpreadsheet,
  Trash2,
  Edit3,
  ExternalLink,
  Plus,
  AlertTriangle,
  X,
  CheckCircle2,
  Download,
  Eye,
  Search,
  History,
  ShieldCheck,
  ChevronLeft,
  ChevronRight
} from 'lucide-react';

export const CaseHistory: React.FC = () => {
  const navigate = useNavigate();

  const [activeTab, setActiveTab] = useState<'cases' | 'audit'>('cases');
  const [cases, setCases] = useState<CaseSummary[]>([]);
  const [auditEvents, setAuditEvents] = useState<AuditEvent[]>([]);
  const [loading, setLoading] = useState(true);

  // Edit Modal State
  const [editingCase, setEditingCase] = useState<CaseSummary | null>(null);
  const [editName, setEditName] = useState('');
  const [editNotes, setEditNotes] = useState('');
  const [savingEdit, setSavingEdit] = useState(false);

  // Delete Confirm State
  const [deletingCase, setDeletingCase] = useState<CaseSummary | null>(null);
  const [deleting, setDeleting] = useState(false);

  // Interactive CSV Dataset Viewer Modal State (Requirement 5)
  const [previewCase, setPreviewCase] = useState<CaseDetailType | null>(null);
  const [loadingPreview, setLoadingPreview] = useState(false);
  const [csvSearchTerm, setCsvSearchTerm] = useState('');
  const [csvPage, setCsvPage] = useState(1);
  const pageSize = 10;

  const [toastMsg, setToastMsg] = useState<string | null>(null);

  useEffect(() => {
    loadData();
  }, [activeTab]);

  const loadData = async () => {
    try {
      setLoading(true);
      if (activeTab === 'cases') {
        const data = await api.getCases();
        setCases(data);
      } else {
        const events = await api.getAuditEvents(50);
        setAuditEvents(events);
      }
    } catch (err) {
      console.error(err);
    } finally {
      setLoading(false);
    }
  };

  const handleOpenDatasetViewer = async (caseId: number) => {
    try {
      setLoadingPreview(true);
      setCsvSearchTerm('');
      setCsvPage(1);
      const detail = await api.getCaseDetail(caseId);
      setPreviewCase(detail);
    } catch (err: any) {
      setToastMsg(`Failed to load dataset: ${err.message}`);
      setTimeout(() => setToastMsg(null), 3000);
    } finally {
      setLoadingPreview(false);
    }
  };

  const handleOpenEdit = (c: CaseSummary) => {
    setEditingCase(c);
    setEditName(c.case_name);
    setEditNotes(c.notes || '');
  };

  const handleSaveEdit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!editingCase) return;
    setSavingEdit(true);
    try {
      await api.updateCase(editingCase.id, editName, editNotes);
      setEditingCase(null);
      setToastMsg('Case metadata updated successfully.');
      setTimeout(() => setToastMsg(null), 3000);
      loadData();
    } catch (err) {
      console.error(err);
    } finally {
      setSavingEdit(false);
    }
  };

  const handleConfirmDelete = async () => {
    if (!deletingCase) return;
    setDeleting(true);
    try {
      await api.deleteCase(deletingCase.id);
      setDeletingCase(null);
      setToastMsg('Case and associated telemetry deleted successfully.');
      setTimeout(() => setToastMsg(null), 3000);
      loadData();
    } catch (err) {
      console.error(err);
    } finally {
      setDeleting(false);
    }
  };

  const handleInspectLoanFromCsv = (loanId: any) => {
    setPreviewCase(null);
    navigate(`/loan-detail?loan_id=${loanId}`);
  };

  // CSV Filter & Pagination calculations
  const rawRows = previewCase?.raw_csv_data || [];
  const csvHeaders = rawRows.length > 0 ? Object.keys(rawRows[0]) : [];
  const filteredRows = rawRows.filter((r) => {
    if (!csvSearchTerm.trim()) return true;
    const term = csvSearchTerm.toLowerCase();
    return Object.values(r).some((val) => String(val).toLowerCase().includes(term));
  });
  const totalPages = Math.ceil(filteredRows.length / pageSize) || 1;
  const paginatedRows = filteredRows.slice((csvPage - 1) * pageSize, csvPage * pageSize);

  return (
    <div className="max-w-[1560px] mx-auto px-4 sm:px-6 lg:px-8 py-6 space-y-6">
      {toastMsg && (
        <div className="fixed top-20 right-8 z-50 bg-espresso text-white px-5 py-3 rounded-2xl shadow-xl flex items-center gap-3 border border-amberGlow/40">
          <CheckCircle2 className="w-5 h-5 text-amberGlow shrink-0" />
          <span className="text-xs font-semibold">{toastMsg}</span>
        </div>
      )}

      {/* Header */}
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4">
        <div>
          <div className="flex items-center gap-2 mb-1">
            <span className="px-2.5 py-0.5 rounded-full bg-amberGlow/15 text-cognac text-[11px] font-bold tracking-wide uppercase border border-amberGlow/30">
              Auditable Telemetry Archives
            </span>
            <span className="text-outline text-xs">• Historical Registry & Surveillance</span>
          </div>
          <h1 className="text-2xl sm:text-3xl font-extrabold text-espresso tracking-tight">
            Case & Dataset Management (CRUD)
          </h1>
          <p className="text-xs sm:text-sm text-outline mt-1">
            Inspect raw CSV datasets, view prediction summaries, and manage supervisory audit trails.
          </p>
        </div>

        <div className="flex items-center gap-2">
          {/* View Tab Selector */}
          <div className="flex items-center bg-surface-container-lowest p-1 rounded-full border border-sandBorder text-xs font-bold text-espresso">
            <button
              onClick={() => setActiveTab('cases')}
              className={`px-3.5 py-1.5 rounded-full transition-all ${
                activeTab === 'cases' ? 'bg-espresso text-white shadow-xs' : 'text-outline hover:text-espresso'
              }`}
            >
              Evaluation Cases ({cases.length})
            </button>
            <button
              onClick={() => setActiveTab('audit')}
              className={`px-3.5 py-1.5 rounded-full transition-all flex items-center gap-1.5 ${
                activeTab === 'audit' ? 'bg-espresso text-white shadow-xs' : 'text-outline hover:text-espresso'
              }`}
            >
              <History className="w-3.5 h-3.5" />
              <span>Supervisory Audit Trail</span>
            </button>
          </div>

          <NavLink
            to="/upload"
            className="flex items-center gap-2 px-4 py-2 rounded-full bg-espresso hover:bg-stone-800 text-white text-xs font-bold transition-all shadow-md"
          >
            <Plus className="w-4 h-4 text-primary-container" />
            <span>New Examination Case</span>
          </NavLink>
        </div>
      </div>

      {/* Main Content: Cases Table or Audit Trail */}
      {activeTab === 'cases' ? (
        <div className="bg-surface-container-lowest rounded-3xl p-6 border border-sandBorder shadow-warm">
          {loading ? (
            <div className="py-12 text-center text-xs text-outline">Loading cases...</div>
          ) : cases.length === 0 ? (
            <div className="py-12 text-center text-xs text-outline">
              No examination cases found. Click "New Examination Case" to upload a portfolio.
            </div>
          ) : (
            <div className="overflow-x-auto rounded-2xl border border-sandBorder">
              <table className="w-full text-left border-collapse text-xs">
                <thead>
                  <tr className="bg-surface-container border-b border-sandBorder text-outline uppercase tracking-wider text-[11px] font-bold">
                    <th className="py-3 px-4">Case Name</th>
                    <th className="py-3 px-4">Upload Date</th>
                    <th className="py-3 px-4">Interactive Dataset File</th>
                    <th className="py-3 px-4">Facilities</th>
                    <th className="py-3 px-4">Risk Result</th>
                    <th className="py-3 px-4">Status</th>
                    <th className="py-3 px-4 text-right">Actions</th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-sandBorder bg-white">
                  {cases.map((c) => (
                    <tr key={c.id} className="hover:bg-[#FFF9EE] transition-colors">
                      <td className="py-3.5 px-4">
                        <NavLink
                          to={`/cases/${c.id}`}
                          className="font-bold text-espresso hover:text-cognac transition-colors block text-sm"
                        >
                          {c.case_name}
                        </NavLink>
                        <span className="text-[11px] text-outline line-clamp-1">{c.notes || 'No description'}</span>
                      </td>
                      <td className="py-3.5 px-4 text-outline font-mono text-[11px]">
                        {new Date(c.created_at).toLocaleDateString()}
                      </td>
                      {/* Interactive CSV link (Requirement 5) */}
                      <td className="py-3.5 px-4 font-mono">
                        <button
                          onClick={() => handleOpenDatasetViewer(c.id)}
                          title="Click to preview dataset table"
                          className="inline-flex items-center gap-1.5 px-2.5 py-1 rounded-lg bg-surface-container hover:bg-amber-100 hover:border-amber-300 border border-sandBorder text-espresso font-semibold text-[11px] transition-all group"
                        >
                          <FileSpreadsheet className="w-3.5 h-3.5 text-cognac group-hover:scale-110 transition-transform" />
                          <span>{c.original_filename}</span>
                          <Eye className="w-3 h-3 text-outline group-hover:text-espresso ml-1" />
                        </button>
                      </td>
                      <td className="py-3.5 px-4 font-bold text-espresso font-mono">
                        {c.total_facilities}
                      </td>
                      <td className="py-3.5 px-4">
                        <span
                          className={`inline-flex items-center gap-1 px-2.5 py-0.5 rounded-full font-bold text-[11px] ${
                            c.critical_count > 0 ? 'bg-red-100 text-error' : 'bg-emerald-100 text-tertiary'
                          }`}
                        >
                          ⚡ {(c.avg_risk_6m * 100).toFixed(1)}% Mean ({c.critical_count} Critical)
                        </span>
                      </td>
                      <td className="py-3.5 px-4">
                        <span className="px-2.5 py-0.5 rounded-full bg-emerald-50 text-tertiary border border-emerald-200 text-[10px] font-bold">
                          {c.status}
                        </span>
                      </td>
                      <td className="py-3.5 px-4 text-right">
                        <div className="flex items-center justify-end gap-1.5">
                          <button
                            onClick={() => handleOpenDatasetViewer(c.id)}
                            title="Inspect Dataset Raw Table"
                            className="p-1.5 rounded-lg bg-surface-container hover:bg-amber-100 text-espresso transition-colors"
                          >
                            <FileSpreadsheet className="w-3.5 h-3.5 text-cognac" />
                          </button>
                          <NavLink
                            to={`/cases/${c.id}`}
                            title="View Case Analysis"
                            className="p-1.5 rounded-lg bg-surface-container hover:bg-espresso hover:text-white text-espresso transition-colors"
                          >
                            <ExternalLink className="w-3.5 h-3.5" />
                          </NavLink>
                          <button
                            onClick={() => handleOpenEdit(c)}
                            title="Edit Case Details"
                            className="p-1.5 rounded-lg bg-surface-container hover:bg-stone-200 text-espresso transition-colors"
                          >
                            <Edit3 className="w-3.5 h-3.5" />
                          </button>
                          <button
                            onClick={() => setDeletingCase(c)}
                            title="Delete Case"
                            className="p-1.5 rounded-lg bg-surface-container hover:bg-red-100 hover:text-error text-outline transition-colors"
                          >
                            <Trash2 className="w-3.5 h-3.5" />
                          </button>
                        </div>
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          )}
        </div>
      ) : (
        /* Supervisory Audit Trail View (Requirement 22) */
        <div className="bg-surface-container-lowest rounded-3xl p-6 border border-sandBorder shadow-warm space-y-4">
          <div className="flex items-center justify-between pb-2 border-b border-sandBorder">
            <div>
              <h2 className="text-base font-bold text-espresso tracking-tight">
                Supervisory Governance & Action Audit Trail
              </h2>
              <p className="text-xs text-outline mt-0.5">
                Immutable ledger of analyst evaluations, watchlist additions, facility escalations, and model telemetry events.
              </p>
            </div>
            <span className="px-3 py-1 rounded-full bg-surface-container text-espresso text-xs font-bold font-mono">
              {auditEvents.length} Recorded Events
            </span>
          </div>

          <div className="overflow-x-auto rounded-2xl border border-sandBorder">
            <table className="w-full text-left border-collapse text-xs">
              <thead>
                <tr className="bg-surface-container border-b border-sandBorder text-outline uppercase tracking-wider text-[11px] font-bold">
                  <th className="py-3 px-4">Timestamp</th>
                  <th className="py-3 px-4">User</th>
                  <th className="py-3 px-4">Action</th>
                  <th className="py-3 px-4">Event Details</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-sandBorder bg-white font-mono text-[11px]">
                {auditEvents.map((ev) => (
                  <tr key={ev.id} className="hover:bg-[#FFF9EE] transition-colors">
                    <td className="py-3 px-4 text-outline whitespace-nowrap">
                      {new Date(ev.created_at).toLocaleString()}
                    </td>
                    <td className="py-3 px-4 font-bold text-espresso whitespace-nowrap">
                      @{ev.username}
                    </td>
                    <td className="py-3 px-4 whitespace-nowrap">
                      <span className="px-2 py-0.5 rounded-md bg-surface-container text-espresso font-sans font-bold text-[10px]">
                        {ev.action}
                      </span>
                    </td>
                    <td className="py-3 px-4 text-stone-700 font-sans max-w-lg">
                      {ev.details}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      )}

      {/* Interactive CSV Dataset Viewer Modal (Requirement 5) */}
      {previewCase && (
        <div className="fixed inset-0 z-50 bg-black/60 backdrop-blur-xs flex items-center justify-center p-4">
          <div className="w-full max-w-5xl bg-white rounded-3xl p-6 border border-sandBorder shadow-2xl space-y-4 max-h-[90vh] flex flex-col">
            <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-3 border-b border-sandBorder pb-3">
              <div>
                <div className="flex items-center gap-2">
                  <FileSpreadsheet className="w-5 h-5 text-cognac" />
                  <h3 className="text-base font-bold text-espresso">{previewCase.original_filename}</h3>
                  <span className="px-2.5 py-0.5 rounded-full bg-surface-container text-outline text-[11px] font-mono">
                    {rawRows.length} facilities / rows
                  </span>
                </div>
                <p className="text-xs text-outline mt-0.5">
                  Raw tabular dataset view. Click "Inspect" on any facility row to jump to the Loan Detail page.
                </p>
              </div>

              <div className="flex items-center gap-2">
                <a
                  href={api.downloadCaseCsvUrl(previewCase.id)}
                  download={previewCase.original_filename}
                  className="px-3 py-1.5 rounded-full bg-surface-container hover:bg-surface-container-high text-espresso text-xs font-bold flex items-center gap-1.5 transition-colors"
                >
                  <Download className="w-3.5 h-3.5" />
                  <span>Download CSV</span>
                </a>

                <button
                  type="button"
                  onClick={() => setPreviewCase(null)}
                  className="w-8 h-8 rounded-full bg-surface-container hover:bg-stone-200 flex items-center justify-center text-outline"
                >
                  <X className="w-4 h-4" />
                </button>
              </div>
            </div>

            {/* Search and Navigation Bar inside Modal */}
            <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-3">
              <div className="relative w-full max-w-xs">
                <Search className="w-4 h-4 text-outline absolute left-3 top-2.5" />
                <input
                  type="text"
                  value={csvSearchTerm}
                  onChange={(e) => {
                    setCsvSearchTerm(e.target.value);
                    setCsvPage(1);
                  }}
                  placeholder="Filter rows by ID, borrower, sector..."
                  className="w-full pl-9 pr-3 py-1.5 rounded-xl bg-surface-container-low border border-sandBorder text-xs text-espresso focus:outline-none focus:ring-1 focus:ring-cognac"
                />
              </div>

              {/* Pagination Controls */}
              <div className="flex items-center gap-2 text-xs font-bold text-espresso">
                <span className="text-outline text-[11px]">
                  Showing {(csvPage - 1) * pageSize + 1} - {Math.min(csvPage * pageSize, filteredRows.length)} of {filteredRows.length}
                </span>
                <button
                  disabled={csvPage <= 1}
                  onClick={() => setCsvPage(csvPage - 1)}
                  className="p-1 rounded-lg bg-surface-container disabled:opacity-40"
                >
                  <ChevronLeft className="w-4 h-4" />
                </button>
                <span className="px-2 py-0.5 rounded bg-surface-container-low font-mono">
                  {csvPage} / {totalPages}
                </span>
                <button
                  disabled={csvPage >= totalPages}
                  onClick={() => setCsvPage(csvPage + 1)}
                  className="p-1 rounded-lg bg-surface-container disabled:opacity-40"
                >
                  <ChevronRight className="w-4 h-4" />
                </button>
              </div>
            </div>

            {/* Table Area */}
            <div className="overflow-x-auto rounded-2xl border border-sandBorder flex-1 max-h-[500px]">
              <table className="w-full text-left border-collapse text-xs font-mono">
                <thead className="sticky top-0 bg-surface-container z-10">
                  <tr className="border-b border-sandBorder text-outline uppercase tracking-wider text-[10px] font-bold font-sans">
                    <th className="py-2.5 px-3">Inspect</th>
                    {csvHeaders.map((h, i) => (
                      <th key={i} className="py-2.5 px-3 whitespace-nowrap">
                        {h}
                      </th>
                    ))}
                  </tr>
                </thead>
                <tbody className="divide-y divide-sandBorder bg-white text-[11px]">
                  {paginatedRows.map((row, rIdx) => {
                    const loanIdCand = row.loan_id || row.facility_id || row.id || row.loan;
                    return (
                      <tr key={rIdx} className="hover:bg-[#FFF9EE] transition-colors">
                        <td className="py-2 px-3 whitespace-nowrap">
                          {loanIdCand ? (
                            <button
                              onClick={() => handleInspectLoanFromCsv(loanIdCand)}
                              className="px-2.5 py-0.5 rounded-full bg-espresso text-white text-[10px] font-bold hover:bg-stone-800 transition-colors"
                            >
                              Inspect
                            </button>
                          ) : (
                            <span className="text-outline text-[10px]">—</span>
                          )}
                        </td>
                        {csvHeaders.map((h, cIdx) => (
                          <td key={cIdx} className="py-2 px-3 whitespace-nowrap text-stone-800">
                            {String(row[h] ?? '')}
                          </td>
                        ))}
                      </tr>
                    );
                  })}
                </tbody>
              </table>
            </div>
          </div>
        </div>
      )}

      {/* Edit Modal (CRUD: Update) */}
      {editingCase && (
        <div className="fixed inset-0 z-50 bg-black/50 backdrop-blur-xs flex items-center justify-center p-4">
          <form
            onSubmit={handleSaveEdit}
            className="w-full max-w-md bg-white rounded-3xl p-6 border border-sandBorder shadow-2xl space-y-4"
          >
            <div className="flex items-center justify-between border-b border-sandBorder pb-3">
              <h3 className="text-base font-bold text-espresso">Edit Case Metadata</h3>
              <button
                type="button"
                onClick={() => setEditingCase(null)}
                className="w-7 h-7 rounded-full bg-surface-container flex items-center justify-center text-outline"
              >
                <X className="w-4 h-4" />
              </button>
            </div>

            <div>
              <label className="block text-xs font-bold text-espresso mb-1">Case Name</label>
              <input
                type="text"
                required
                value={editName}
                onChange={(e) => setEditName(e.target.value)}
                className="w-full px-3.5 py-2 rounded-xl border border-sandBorder text-xs text-espresso focus:outline-none focus:ring-2 focus:ring-primary-container"
              />
            </div>

            <div>
              <label className="block text-xs font-bold text-espresso mb-1">Examination Notes</label>
              <textarea
                rows={3}
                value={editNotes}
                onChange={(e) => setEditNotes(e.target.value)}
                className="w-full px-3.5 py-2 rounded-xl border border-sandBorder text-xs text-espresso focus:outline-none focus:ring-2 focus:ring-primary-container"
              ></textarea>
            </div>

            <div className="flex items-center justify-end gap-2 pt-2">
              <button
                type="button"
                onClick={() => setEditingCase(null)}
                className="px-4 py-2 rounded-full bg-surface-container text-xs font-bold text-outline hover:text-espresso"
              >
                Cancel
              </button>
              <button
                type="submit"
                disabled={savingEdit}
                className="px-5 py-2 rounded-full bg-espresso text-white text-xs font-bold hover:bg-stone-800 disabled:opacity-50"
              >
                {savingEdit ? 'Saving...' : 'Save Changes'}
              </button>
            </div>
          </form>
        </div>
      )}

      {/* Delete Confirmation Modal (CRUD: Delete) */}
      {deletingCase && (
        <div className="fixed inset-0 z-50 bg-black/50 backdrop-blur-xs flex items-center justify-center p-4">
          <div className="w-full max-w-sm bg-white rounded-3xl p-6 border border-sandBorder shadow-2xl space-y-4 text-center">
            <div className="w-12 h-12 rounded-full bg-red-100 text-error flex items-center justify-center mx-auto">
              <AlertTriangle className="w-6 h-6" />
            </div>
            <div>
              <h3 className="text-base font-bold text-espresso">Delete Examination Case?</h3>
              <p className="text-xs text-outline mt-1">
                Are you sure you want to delete <strong className="text-espresso">{deletingCase.case_name}</strong>?
                This will permanently delete the uploaded dataset, calculated risk vectors, and prediction telemetry.
              </p>
            </div>

            <div className="flex items-center justify-center gap-2 pt-2">
              <button
                onClick={() => setDeletingCase(null)}
                className="px-4 py-2 rounded-full bg-surface-container text-xs font-bold text-outline hover:text-espresso"
              >
                Cancel
              </button>
              <button
                onClick={handleConfirmDelete}
                disabled={deleting}
                className="px-5 py-2 rounded-full bg-red-600 hover:bg-red-700 text-white text-xs font-bold disabled:opacity-50"
              >
                {deleting ? 'Deleting...' : 'Confirm Delete'}
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};
