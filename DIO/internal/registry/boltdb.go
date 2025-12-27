package registry

import (
	"encoding/json"
	"time"

	"github.com/boltdb/bolt"
	pb "github.com/nisaral/dio/api/proto"
)

// Store wraps our BoltDB connection
type Store struct {
	db *bolt.DB
}

// NewStore initializes the BoltDB file and creates the "workers" bucket
func NewStore(dbPath string) (*Store, error) {
	db, err := bolt.Open(dbPath, 0600, &bolt.Options{Timeout: 1 * time.Second})
	if err != nil {
		return nil, err
	}

	err = db.Update(func(tx *bolt.Tx) error {
		_, err := tx.CreateBucketIfNotExists([]byte("workers"))
		return err
	})

	return &Store{db: db}, err
}

// SaveWorker stores or updates a worker's registration details
func (s *Store) SaveWorker(req *pb.RegisterRequest) error {
	return s.db.Update(func(tx *bolt.Tx) error {
		b := tx.Bucket([]byte("workers"))

		// We serialize the Protobuf message to JSON to store it in Bolt
		data, err := json.Marshal(req)
		if err != nil {
			return err
		}

		return b.Put([]byte(req.WorkerId), data)
	})
}

// ListWorkers retrieves all registered workers for the scheduler
func (s *Store) ListWorkers() ([]*pb.RegisterRequest, error) {
	var workers []*pb.RegisterRequest
	err := s.db.View(func(tx *bolt.Tx) error {
		b := tx.Bucket([]byte("workers"))
		return b.ForEach(func(k, v []byte) error {
			var w pb.RegisterRequest
			if err := json.Unmarshal(v, &w); err != nil {
				return err
			}
			workers = append(workers, &w)
			return nil
		})
	})
	return workers, err
}
