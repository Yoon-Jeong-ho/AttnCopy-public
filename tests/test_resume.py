import unittest
from scripts.resume_pilot import recovery_plan,validate_rows,merge_rows,ARMS


class ResumeTests(unittest.TestCase):
    def test_snapshot_missing_order_and_merge(self):
        records,old,todo,*_ = recovery_plan()
        self.assertEqual(len(todo),528)
        by_id = {r['id']:r for r in old}
        new = []
        for i,arm in todo:
            template = old[0]
            new.append(dict(template,id=records[i]['id'],arm=arm,example_index=i,
                       order=ARMS[i%5:]+ARMS[:i%5],split='test'))
        merged=merge_rows(old,new,records)
        self.assertEqual(len(merged),800)
        self.assertEqual(sum(r['latency_boundary_excluded'] for r in merged),5)
        self.assertEqual(sum(r['source_segment']=='segment1' for r in merged),272)
        with self.assertRaises(AssertionError):
            validate_rows(old+[old[0]],records)


if __name__=='__main__':
    unittest.main()
